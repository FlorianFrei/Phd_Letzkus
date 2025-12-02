load_data_with_brain_alignment <- function(datapath, metapath, depthpath, structure_path) {
  library(data.table)
  
  # --- depth info
  depth_json <- jsonlite::fromJSON(depthpath, simplifyMatrix = TRUE, flatten = TRUE)
  depth <- data.table::rbindlist(depth_json[1:(length(depth_json) - 1)], fill = TRUE) %>%
    mutate(ch = row_number())
  
  # --- structure tree
  st <- fread(structure_path) %>%
    select(id, name, acronym, depth, structure_id_path, parent_structure_id)
  
  depth <- depth %>%
    left_join(st, by = c("brain_region" = "acronym"))
  
  # --- define areas (case_when for readability)
  HUAC <- st %>%
    mutate(area = case_when(
      str_detect(structure_id_path, "/247/|/541/") ~ "AUD",
      str_detect(structure_id_path, "/131/|/295/|/319/|/780/") ~ "AMY",
      str_detect(structure_id_path, "/1089/") ~ "HIP",
      str_detect(structure_id_path, "/1008/") ~ "MG",
      str_detect(structure_id_path, "/549/") ~ "TH",
      str_detect(structure_id_path, "/315/") ~ "Cortex",
      TRUE ~ "undefined"
    )) %>%
    filter(!(area == "Cortex" & structure_id_path %in% structure_id_path[area == "AUD"])) %>%
    filter(!(area == "TH" & structure_id_path %in% structure_id_path[area == "MG"])) %>%
    select(area, structure_id_path)
  
  depth <- depth %>%
    left_join(HUAC, by = "structure_id_path") %>%
    select(brain_region, area, ch)
  
  # --- cluster meta
  meta <- fread(metapath) %>%
    left_join(depth, by = "ch") %>%
    filter(bc_unitType == "GOOD" | bc_unitType=='MUA') %>%
    select(cluster_id, brain_region, area, depth)
  
  # --- main data
  data <- fread(datapath) %>%
    filter(cluster_id %in% meta$cluster_id) %>%
    left_join(meta, by = "cluster_id") %>%
    mutate(state = ifelse(state_name %in% c("Downsweep", "Opto_Upwsweep", "Opto_Downsweep", "Upsweep"), "Sound", state_name)) %>%mutate(
  trial_type = as.character(trial_type),
  trial_type = case_when(
    trial_type == '1' ~ 'Upsweep',
    trial_type == '2' ~ 'Downsweep',
    trial_type == '3' ~ 'Opto_Upwsweep',
    trial_type == '4' ~ 'Opto_Downsweep',
    TRUE ~ trial_type  # keep anything else as-is
  )
) %>% select(!state_name)

  data
}

load_data <- function(datapath, metapath) {
  
  library(data.table)
  
  
  # --- cluster meta
  meta <- fread(metapath)  %>%
    select(cluster_id, depth)
  
  # --- main data
  data <- fread(datapath) %>%
    filter(cluster_id %in% meta$cluster_id) %>%
    left_join(meta, by = "cluster_id") %>%
    mutate(state = ifelse(state_name %in% c("Downsweep", "Opto_Upwsweep", "Opto_Downsweep", "Upsweep"), "Sound", state_name)) %>% mutate(trial_type = case_when(
      trial_type ==1 ~ 'Upsweep',
      trial_type ==2 ~ 'Downsweep',
      trial_type ==3 ~ 'Opto_Upwsweep',
      trial_type ==4 ~ 'Opto_Downsweep',
    )) %>% select(!state_name)
  
  data
}


Align_to_state <- function(Data, Alignment_State) {
  filter2 <- Data %>%
    group_by(trial_number) %>%
    filter(state == Alignment_State) %>%
    slice_min(time_bin) %>%
    mutate(minT = time_bin) %>%
    select(minT, trial_number) %>%
    distinct()
  Data <-Data %>% left_join(filter2)
  
  nearest_fixed <- function(times, fixed_times) {
    i <- 1
    out <- numeric(length(times))
    
    for (j in seq_along(times)) {
      # move pointer forward while the next fixed time is closer
      while (i < length(fixed_times) &&
             abs(fixed_times[i + 1] - times[j]) < abs(fixed_times[i] - times[j])) {
        i <- i + 1
      }
      out[j] <- fixed_times[i]
    }
    out
  }
  Data <- Data %>% 
  group_by(Animal, condition) %>% 
  arrange(Animal, condition, time_bin) %>%  # CRITICAL: sort time_bin first
  mutate(
    # Get the sorted unique minT values for this Animal/condition
    nearest_fixed = nearest_fixed(time_bin, sort(unique(minT[!is.na(minT)]))),
    rel_time = round(time_bin - nearest_fixed, 2)
  ) %>%
  ungroup()
  
  
 Data <- Data %>%
  arrange(Animal, condition, time_bin) %>%  # explicit multi-column sort
  group_by(Animal, condition) %>%
  mutate(aligntrial = cumsum(c(TRUE, diff(rel_time) < -1))) %>%
  unite('aligntrial2', aligntrial, Animal, condition, remove = FALSE) %>%
  select(-aligntrial) %>%
  rename(aligntrial = aligntrial2)



  Data <- Data %>%
    group_by(aligntrial) %>%
    mutate(trial_type = first(trial_type)) %>%
    ungroup()
  

  
  datarange <- Data %>%
    group_by(aligntrial) %>%
    reframe(range = range(rel_time)) %>%
    group_by() %>%
    summarise(
      common_min = max(range[c(TRUE, FALSE)]), # max of minimums
      common_max = min(range[c(FALSE, TRUE)]) # min of maximums
    )
  print(paste("common rel_time:", datarange$common_min, datarange$common_max))
  Data
}


safe_load_data <- function(datapath, metapath, depthpath, structure_path) {
  if (is.na(depthpath) || !file.exists(depthpath)) {
    message("⚠️ No depth file for: ", basename(datapath))
    
    # fallback: use load_data (without depth)
    data <- load_data(datapath, metapath) %>%
      mutate(file_id = file_path_sans_ext(basename(datapath)))
    
  } else {
    # use depth-aligned version
    data <- load_data_with_brain_alignment(datapath, metapath, depthpath, structure_path) %>%
      mutate(file_id = file_path_sans_ext(basename(datapath)))
  }
  data
}


process_all_data <- function(datapath,
                             metapath,
                             depthpath,
                             Alignment_State,
                             structure_path = r"(C:\Users\Freitag\Documents\GitHub\Phd_Letzkus\pre_processing\Allenbrain_structure_tree.csv)",
                             out_dir = NULL) {
  library(tidyverse)
  library(tools)
  source("helpers.R")
  
  # --- determine output directory
  if (is.null(out_dir)) {
    out_dir <- file.path(datapath, "processed_chunks")
  }
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  
  # --- list input files
  list_of_data   <- list.files(datapath, pattern = "\\.csv$", full.names = TRUE)
  list_of_meta   <- list.files(metapath, pattern = "\\.tsv$", full.names = TRUE)
  list_of_depths <- list.files(depthpath, pattern = "\\.json$", full.names = TRUE)
  
  # --- prepare matching table
  data_ids  <- file_path_sans_ext(basename(list_of_data))
  meta_ids  <- file_path_sans_ext(basename(list_of_meta))
  depth_ids <- file_path_sans_ext(basename(list_of_depths))
  
  lookup <- tibble(
    data  = list_of_data,
    meta  = map_chr(data_ids, ~ list_of_meta[match(.x, meta_ids)]),
    depth = map_chr(data_ids, ~ {
      matched <- list_of_depths[match(.x, depth_ids)]
      ifelse(is.na(matched), NA_character_, matched)
    })
  )
  
  # --- define Align_to_state inline
  Align_to_state <- function(Data, Alignment_State) {
    filter2 <- Data %>%
      group_by(trial_number) %>%
      filter(state == Alignment_State) %>%
      slice_min(time_bin) %>%
      mutate(minT = time_bin) %>%
      select(minT, trial_number) %>%
      distinct()
    Data <- Data %>% left_join(filter2, by = "trial_number")
    
    nearest_fixed <- function(times, fixed_times) {
      i <- 1
      out <- numeric(length(times))
      for (j in seq_along(times)) {
        while (i < length(fixed_times) &&
               abs(fixed_times[i + 1] - times[j]) < abs(fixed_times[i] - times[j])) {
          i <- i + 1
        }
        out[j] <- fixed_times[i]
      }
      out
    }
    
    Data <- Data %>%
      group_by(Animal, condition) %>%
      arrange(Animal, condition, time_bin) %>%
      mutate(
        nearest_fixed = nearest_fixed(time_bin, sort(unique(minT[!is.na(minT)]))),
        rel_time = round(time_bin - nearest_fixed, 2)
      ) %>%
      ungroup() %>%
      arrange(Animal, condition, time_bin) %>%
      group_by(Animal, condition) %>%
      mutate(aligntrial = cumsum(c(TRUE, diff(rel_time) < -1))) %>%
      unite('aligntrial2', aligntrial, Animal, condition, remove = FALSE) %>%
      select(-aligntrial) %>%
      rename(aligntrial = aligntrial2) %>%
      group_by(aligntrial) %>%
      mutate(trial_type = first(trial_type)) %>%
      ungroup()
    
    datarange <- Data %>%
      group_by(aligntrial) %>%
      reframe(range = range(rel_time)) %>%
      group_by() %>%
      summarise(
        common_min = max(range[c(TRUE, FALSE)]),
        common_max = min(range[c(FALSE, TRUE)])
      )
    message("common rel_time: ", datarange$common_min, " to ", datarange$common_max)
    Data
  }
  
  # --- process each dataset sequentially (low memory)
  walk2(lookup$data, seq_len(nrow(lookup)), \(data_file, i) {
    message("Processing file ", i, " of ", nrow(lookup), ": ", basename(data_file))
    meta_file  <- lookup$meta[i]
    depth_file <- lookup$depth[i]

    # --- load data (with or without depth alignment)
    d <- if (is.na(depth_file) || !file.exists(depth_file)) {
      message("⚠️ No depth file for: ", basename(data_file))
      load_data(data_file, meta_file) %>%
        mutate(file_id = file_path_sans_ext(basename(data_file)))
    } else {
      load_data_with_brain_alignment(data_file, meta_file, depth_file, structure_path) %>%
        mutate(file_id = file_path_sans_ext(basename(data_file)))
    }

    # --- tidy postprocessing
    d <- d %>%
      mutate(file_id2 = file_id,
             file_id3 = file_id) %>%
      unite('cluster_id', file_id, cluster_id, remove = FALSE) %>%
      unite('trial_number', file_id2, trial_number, remove = FALSE) %>%
      separate_wider_delim(file_id3, names = c('Animal', 'condition'), delim = '_')

    # --- align per chunk
    d <- Align_to_state(d, Alignment_State = Alignment_State)

    # --- save chunk
    out_file <- file.path(out_dir, paste0(basename(data_file), "_processed.csv"))
    readr::write_csv(d, out_file)
    message("✅ Saved: ", basename(out_file), "  (", nrow(d), " rows)")

    rm(d); gc()
  })
  
  # --- recombine all processed chunks
  message("Combining processed chunks ...")
  processed_files <- list.files(out_dir, pattern = "_processed\\.csv$", full.names = TRUE)
  data <- map_dfr(processed_files, read_csv, show_col_types = FALSE)
  
  message("✅ Done. Combined data has ", nrow(data), " rows.")
  return(data)
}



align_trials <- function(data,baseline_corr = T) {
  
  meta_df <- data %>%
  select(cluster_id, brain_region:depth, Animal, condition) %>%
  distinct()

  Trialless <- data %>%
  group_by(trial_type, rel_time, cluster_id) %>%
  summarise(total_event_count = sum(event_count), .groups = "drop") %>%
  left_join(meta_df, by = "cluster_id") %>%
  left_join(
    data %>%
      group_by(Animal, condition, trial_type) %>%
      summarise(n_trials = n_distinct(trial_number), .groups = "drop"),
    by = c("Animal", "condition", "trial_type")
  ) %>%
  mutate(rate = total_event_count / n_trials / 0.01)
  
  if (baseline_corr == T) {
    baseline <- Trialless %>%
  filter(between(rel_time, -15, -5)) %>%
  group_by(cluster_id) %>%
  summarise(cl_mean = mean(rate), cl_sd = sd(rate))
  Trialless <- Trialless %>%
  left_join(baseline) %>%
  group_by(cluster_id) %>%
  mutate(Zscore = (rate - cl_mean) / cl_sd) %>%
  ungroup() %>% replace_na(list(Zscore=0.001))
    }
  Trialless
}

library(dplyr)
library(tidyr)

align_trials_multiple_files <- function(folder_path, baseline_corr = TRUE) {
  
  # Get the list of all files in the folder
  file_list <- list.files(folder_path, full.names = TRUE)
  
  # Initialize an empty list to store results
  all_results <- list()

  # Process each file in the folder
  for (file in file_list) {
    # Read the file (assuming CSV format; change as needed)
    data <- read.csv(file)
    
    # Metadata selection
    meta_df <- data %>%
      select(cluster_id, brain_region,area, depth, Animal, condition) %>%
      distinct()

    # Compute trialless data
    Trialless <- data %>%
      group_by(trial_type, rel_time, cluster_id) %>%
      summarise(total_event_count = sum(event_count), .groups = "drop") %>%
      left_join(meta_df) %>%
      left_join(
        data %>%
          group_by(Animal, condition, trial_type) %>%
          summarise(n_trials = n_distinct(trial_number), .groups = "drop"),
        by = c("Animal", "condition", "trial_type")
      ) %>%
      mutate(rate = total_event_count / n_trials / 0.01)
    
    # Baseline correction if enabled
    if (baseline_corr) {
      baseline <- Trialless %>%
        filter(between(rel_time, -15, -5)) %>%
        group_by(cluster_id) %>%
        summarise(cl_mean = mean(rate), cl_sd = sd(rate))
      
      Trialless <- Trialless %>%
        left_join(baseline) %>%
        group_by(cluster_id) %>%
        mutate(Zscore = (rate - cl_mean) / cl_sd) %>%
        ungroup() %>%
        replace_na(list(Zscore = 0.001))
    }

    # Append the processed file's result to the list
    message("processed" ,file)
    all_results[[file]] <- Trialless
  }

  # Combine all the results into one data frame
  combined_results <- bind_rows(all_results)

  return(combined_results)
}


load_from_list <- function(list_of_data, list_of_meta) {
  all_loaded <- map2_dfr(list_of_data, list_of_meta, ~ {
    load_data(.x, .y) %>%
    mutate(file_id = tools::file_path_sans_ext(basename(.x)))
  })
  
  data <- all_loaded %>%
    mutate(
      cluster_id = paste(file_id, cluster_id, sep = "_"),
      trial_number = paste(file_id, trial_number, sep = "_")
    ) %>%
    separate(file_id, into = c("Animal", "condition"), sep = "_")
}


asymmetric_smooth <- function(x, window_high = 4, window_low = 25, threshold = 0.25) {
  n <- length(x)
  result <- numeric(n)

  for (i in 1:n) {
    if (x[i] > threshold) {
      window <- window_high
    } else {
      window <- window_low
    }

    start <- max(1, i - window %/% 2)
    end <- min(n, i + window %/% 2)
    result[i] <- mean(x[start:end])
  }
  return(result)
}
