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
    filter(bc_unitType == "GOOD") %>%
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