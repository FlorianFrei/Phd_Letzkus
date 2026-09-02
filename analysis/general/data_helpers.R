# ==============================================================================
# data_helpers.R
# Shared loading, metadata, trial-label, and baseline helpers
# ==============================================================================

library(dplyr)
library(purrr)
library(readr)
library(stringr)
library(tidyr)

# ------------------------------------------------------------------------------
# Small path helper
# ------------------------------------------------------------------------------

file_stem <- function(file_path) {
  tools::file_path_sans_ext(basename(file_path))
}


# ------------------------------------------------------------------------------
# Read every CSV in a folder and add the filename as `source`
#
# Example:
# data <- read_csv_folder("I:/Data/my_folder")
#
# The output receives a `source` column based on each file name:
# "8103_naive.csv" -> source = "8103_naive"
# ------------------------------------------------------------------------------

read_csv_folder <- function(folder_path,
                            recursive = FALSE,
                            source_column = "source",
                            remove_unnamed_columns = TRUE) {
  
  files <- list.files(
    path = folder_path,
    pattern = "\\.csv$",
    recursive = recursive,
    full.names = TRUE
  )
  
  if (length(files) == 0) {
    stop(
      paste("No CSV files found in:", folder_path),
      call. = FALSE
    )
  }
  
  data <- map_dfr(files, function(file) {
    
    one_file <- read_csv(file, show_col_types = FALSE)
    
    if (remove_unnamed_columns) {
      one_file <- one_file %>%
        select(-matches("^\\.\\.\\.\\d+$|^Unnamed:"))
    }
    
    one_file[[source_column]] <- file_stem(file)
    
    one_file
  })
  
  data %>%
    relocate(all_of(source_column))
}


# ------------------------------------------------------------------------------
# Add Animal and condition from `source`
#
# Expected source format:
#   "8103_naive"
#   "8103_recall"
#
# Result:
#   source       Animal  condition
#   8103_naive   8103    naive
#
# Any additional underscores remain in condition:
#   8103_long_recall_session -> Animal = 8103,
#                               condition = long_recall_session
# ------------------------------------------------------------------------------

add_animal_condition <- function(data, source_column = "source") {
  
  source_data <- data %>%
    select(all_of(source_column)) %>%
    separate_wider_delim(
      cols = all_of(source_column),
      delim = "_",
      names = c("Animal", "condition"),
      too_many = "merge",
      too_few = "align_start"
    ) %>%
    mutate(Animal = as.character(Animal))
  
  bind_cols(data, source_data %>% select(Animal, condition))
}


# ------------------------------------------------------------------------------
# Standardize exceptional source names
#
# Keep all source-renaming rules in this one function rather than scattered
# across pupil, speed, and other analysis scripts.
# ------------------------------------------------------------------------------

standardize_source_names <- function(data, source_column = "source") {
  
  data %>%
    mutate(
      "{source_column}" := case_when(
        .data[[source_column]] == "3198-51_g1" ~ "51_naive",
        .data[[source_column]] == "3198-51_recall_g0" ~ "51_recall",
        .data[[source_column]] == "3198-52_g0" ~ "52_naive",
        .data[[source_column]] == "3198-52_recall_g0" ~ "52_recall",
        TRUE ~ as.character(.data[[source_column]])
      )
    )
}


# ------------------------------------------------------------------------------
# Convert numeric/raw BPOD trial types to readable labels
#
# Convention:
# 0 = Opto_only
# 1 = Upsweep
# 2 = Downsweep
# 3 = Opto_Upsweep
# 4 = Opto_Downsweep
# ------------------------------------------------------------------------------

recode_trial_type <- function(data, trial_type_column = "trial_type") {
  
  data %>%
    mutate(
      "{trial_type_column}" := as.character(.data[[trial_type_column]]),
      "{trial_type_column}" := case_when(
        .data[[trial_type_column]] == "0" ~ "Opto_only",
        .data[[trial_type_column]] == "1" ~ "Upsweep",
        .data[[trial_type_column]] == "2" ~ "Downsweep",
        .data[[trial_type_column]] == "3" ~ "Opto_Upsweep",
        .data[[trial_type_column]] == "4" ~ "Opto_Downsweep",
        TRUE ~ .data[[trial_type_column]]
      )
    )
}


# ------------------------------------------------------------------------------
# Load BPOD data and return only sound-trial start times.
#
# Output columns:
# source, Animal, condition, trial_number, continuous_start, trial_type
# ------------------------------------------------------------------------------

load_sound_starts <- function(bpod_path) {
  
  sound_state_names <- c(
    "Downsweep",
    "Upsweep",
    "Opto_Upwsweep",  # included because this spelling appears in raw files
    "Opto_Upsweep",
    "Opto_Downsweep"
  )
  
  read_csv_folder(
    folder_path = bpod_path,
    source_column = "source"
  ) %>%
    mutate(
      state = if_else(
        state_name %in% sound_state_names,
        "Sound",
        state_name
      )
    ) %>%
    recode_trial_type() %>%
    filter(state == "Sound") %>%
    select(-any_of(c("state_name", "state"))) %>%
    standardize_source_names() %>%
    add_animal_condition() %>%
    select(
      source,
      Animal,
      condition,
      trial_number,
      continuous_start,
      trial_type
    )
}


# ------------------------------------------------------------------------------
# Load the shock-direction lookup table.
#
# Output:
# Animal, direction
#
# Direction becomes, for example, "Upsweep" or "Downsweep".
# ------------------------------------------------------------------------------

load_shock_direction <- function(shock_file) {
  
  read_csv(shock_file, show_col_types = FALSE) %>%
    transmute(
      Animal = as.character(Animal2),
      direction = paste0(direction, "sweep")
    )
}


# ------------------------------------------------------------------------------
# Add fear and opto labels to trial-level/timepoint data.
#
# Requires:
# - Animal
# - a stimulus column, normally stim_id
# - shock lookup loaded by load_shock_direction()
#
# Output columns added:
# - direction
# - fear: cs+ / cs-
# - Opto: opto / no
# ------------------------------------------------------------------------------

add_fear_labels <- function(data,
                            shock_lookup,
                            stimulus_column = "stim_id") {
  
  data %>%
    left_join(shock_lookup, by = "Animal") %>%
    mutate(
      fear = if_else(
        direction == str_remove(.data[[stimulus_column]], "^Opto_"),
        "cs+",
        "cs-"
      ),
      Opto = if_else(
        str_starts(.data[[stimulus_column]], "Opto"),
        "opto",
        "no"
      )
    )
}


# ------------------------------------------------------------------------------
# Load cluster quality metrics and retain passing clusters.
#
# Assumes each metrics CSV includes:
# - first column: cluster ID
# - num_spikes
# - presence_ratio
#
# Cluster IDs are returned in the form:
#   original_clusterID_sourceFile
# ------------------------------------------------------------------------------

load_quality_metrics <- function(metrics_path,
                                 presence_cutoff = 0.8) {
  
  metric_files <- list.files(
    path = metrics_path,
    pattern = "\\.csv$",
    full.names = TRUE
  )
  
  if (length(metric_files) == 0) {
    stop(
      paste("No quality-metric CSV files found in:", metrics_path),
      call. = FALSE
    )
  }
  
  map_dfr(metric_files, function(file) {
    
    read_csv(file, show_col_types = FALSE) %>%
      rename(cluster_id = 1) %>%
      select(cluster_id, num_spikes, presence_ratio) %>%
      mutate(source = file_stem(file))
  }) %>%
    filter(presence_ratio > presence_cutoff) %>%
    arrange(num_spikes) %>%
    mutate(
      cluster_id = paste(cluster_id, source, sep = "_")
    ) %>%
    select(-source)
}


# ------------------------------------------------------------------------------
# Calculate a baseline mean, baseline SD, and Z-score.
#
# `value_column` and `group_columns` are character vectors.
# This avoids non-standard evaluation / rlang syntax.
#
# Example: pupil:
#
# df_segments <- add_baseline_zscore(
#   data = df_segments,
#   value_column = "pupil",
#   group_columns = c("source", "trial_sequence"),
#   time_column = "time_bin",
#   baseline_end = -1,
#   zscore_column = "pz"
# )
#
# Example: speed:
#
# df <- add_baseline_zscore(
#   data = df,
#   value_column = "speed",
#   group_columns = c("source", "trial_number"),
#   time_column = "rel_time",
#   baseline_end = -1,
#   zscore_column = "sz"
# )
#
# Calculation is exactly:
# z-score = (value - baseline mean) / baseline SD
#
# Baseline is all rows where time_column < baseline_end.
# ------------------------------------------------------------------------------

add_baseline_zscore <- function(data,
                                value_column,
                                group_columns,
                                time_column,
                                baseline_end = -1,
                                zscore_column = "z") {
  
  required_columns <- c(value_column, group_columns, time_column)
  
  missing_columns <- setdiff(required_columns, names(data))
  
  if (length(missing_columns) > 0) {
    stop(
      paste(
        "These columns are missing from data:",
        paste(missing_columns, collapse = ", ")
      ),
      call. = FALSE
    )
  }
  
  baseline_data <- data %>%
    filter(.data[[time_column]] < baseline_end)
  
  baselines <- baseline_data %>%
    group_by(across(all_of(group_columns))) %>%
    summarise(
      baseline_mean = mean(.data[[value_column]], na.rm = TRUE),
      baseline_sd = sd(.data[[value_column]], na.rm = TRUE),
      .groups = "drop"
    )
  
  data %>%
    left_join(baselines, by = group_columns) %>%
    mutate(
      "{zscore_column}" :=
        (.data[[value_column]] - baseline_mean) / baseline_sd
    )
}