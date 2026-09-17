library(tidyverse)
library(tidybayes)
library(lme4)
library(emmeans)
library(stringr)
library(RColorBrewer)
library(zoo)
library(dplyr)

options(mc.cores = parallel::detectCores())

setwd(dirname(rstudioapi::getActiveDocumentContext()$path)) ### set wd to current dir

res <- read.csv("../../data/langlearning_v2_anonymized.csv") ### anonymized results file (see data/README.md)

bad.subjects <- names(table(res$Results.index)[table(res$Results.index) < 38]) ### remove xiaochen and bob results
res <- res[!(res$Results.index %in% bad.subjects),]

res <- res[,c(1, 3, 7, 10, 12, 13)] ### select relevant columns
colnames(res) <- c("subject", "counter", "controller", "type", "condition", "response") ### rename columns
res <- data.frame(lapply(res, function(x) gsub("%2C", " ", x))) ### change html code %2C to white space

### create data frame with all demographic information
### note: strings are reduced to lower case and leading/trailing white space removed 

demo <- res %>%
  filter(condition %in% c("age", "gender", 
                          "language", "wordOrderBox", "second_language", 
                          "later_language", "notes")) %>%
  select(subject, condition, response) %>%
  mutate(response = trimws(tolower(response))) %>%
  pivot_wider(names_from = condition, values_from = response) %>%
  rename(
    age = age,
    gender = gender,
    firstL = language,
    order = wordOrderBox,
    secondL = second_language,
    laterL = later_language,
    notes = notes
  )

# Make a lookup table so we can translate positions to elements
order_lookup <- c(
  first = "s",
  second = "v",
  third = "o"
)

word_order_df <- res %>%
  filter(condition %in% c("subject", "object", "verb")) %>%
  mutate(
    # map condition to abbreviation
    cond_abbrev = case_when(
      condition == "subject" ~ "s",
      condition == "verb"    ~ "v",
      condition == "object"  ~ "o"
    )
  ) %>%
  # Join position numbers to actual word order
  mutate(order_pos = match(response, c("first", "second", "third"))) %>%
  arrange(subject, order_pos) %>%
  group_by(subject) %>%
  summarise(
    est_order = paste(cond_abbrev[order(order_pos)], collapse = ""),
    .groups = "drop"
  )

# merge into existing demo table
demo <- demo %>%
  left_join(word_order_df, by = "subject")

### do participants have a second language?
demo$secondYN <- ifelse(demo$secondL %in% c("-", "none", "0", "english", "just english",
                                            "n//a", "n/a", "na", "no", "no others", "non",
                                            "none", "none fluently", "none other", "none/na"), 0, 1)

### have participants learned a language later in life?
demo$laterYN <- ifelse(demo$laterL %in% c("-", 0, "english", "n/a", "non", "none",
                                           "none fluently", "none other", "none to any real usefullness",
                                           "none/na", "none2"), 0, 1)

later <- read.csv("lateL.csv", header = FALSE, sep = ";")
second <- read.csv("secondL.csv", header = FALSE, sep = ";")

demo$laterOrder <- later$V2[match(demo$laterL, later$V1)]
demo$secondOrder <- second$V2[match(demo$secondL, second$V1)]

### create results file with which images participants clicked on
df <- res[res$condition == "clickedImage",]
df$response <- trimws(df$response, which = "both")
df <- separate_rows(df, response)

### create data frame with rts for each trial and add to results file
temp <- res[res$condition == "rt",]
temp$rt <- trimws(temp$response, which = "both")
temp <- separate_rows(temp, rt)
df$rt <- as.numeric(paste(temp$rt)) ### ensure values are numeric

df <- df %>%
  group_by(subject) %>%
  mutate(trial = row_number()) %>%
  ungroup()

df$correct <- ifelse(df$response == "target", 1, 0) ### classify response as correct or incorrect

### add order and demographic information to results file
temp <- demo[,c("subject", "order", "secondYN", "laterYN", "age", "gender", "notes")] 
df <- merge(df, temp, by = "subject")
df$age <- as.numeric(paste(df$age))

### check for each pp if performance in the second half was significantly above chance (25%)
### this gives a z value based on a simple one-proportion test
### a z-value of 1.96 (corresponding to p = .05) corresponds to less than 33 correct responses
demo$perf <- sapply(demo$subject, function(x) {
  nrow(df[df$subject == x & df$trial > 100 & df$correct == 1, ])
})

table(demo[demo$perf < 33,]$order)
chisq.test(table(demo[demo$perf < 33,]$order))

### select and remove nonnative speaker
bad.subjects <- demo[demo$firstL == "urdu",]$subject
bad.subjects <- append(bad.subjects, demo[demo$perf < 33,]$subject)
df <- df[!(df$subject %in% bad.subjects),]
demo <- demo[!(demo$subject %in% bad.subjects),]

### ensure both subject and order are viewed as factors
df$subject <- factor(df$subject)
df$order <- toupper(df$order)
df$order <- factor(df$order, levels = c("SVO", "SOV", "VSO", "VOS", "OVS", "OSV"))

df$sqrttrial <- sqrt(df$trial)

### create model predicting correct responses based on the interaction between logtrial and word order
new.model <- glmer(correct ~ 1 + sqrttrial*order + (1 + sqrttrial | subject), 
                   family = binomial(link = "logit"),
                   control = glmerControl(calc.derivs = FALSE),
                   data = df)

### create emtrends model to check for differences between the slopes for word orders
emtrends_obj <- emtrends(new.model, ~ order, var = "sqrttrial")

### show and compare slopes
summary(emtrends_obj)
pairs(emtrends_obj)

### df with observed data for plotting
df_avg <- df %>%
  group_by(order, trial) %>%
  summarise(emp = mean(correct, na.rm = TRUE),
            .groups = "drop")

df_avg$sqrttrial <- sqrt(df_avg$trial)

# generate predicted probabilities based on fixed effects only
df_avg$pred <- predict(new.model,
                       newdata = df_avg,
                       type = "response",
                       re.form = NA)

### plot data and predictions
pdf("learning-data.pdf", height = 3.8, width = 6)
ggplot(data = df) +
  ### plot se ribbon
  geom_smooth(
    aes(x = trial, y = correct, fill = order),
    method = "loess", alpha = 0.3, colour = NA, span = 0.3
  ) +
  ### plot smoothed line
  geom_smooth(
    aes(x = trial, y = correct, colour = order),
    method = "loess", se = FALSE, alpha = 0.9, span = 0.3, size = 1.2
  ) +
  scale_x_continuous(
    limits = c(1, 201),
    expand = c(0, 0.1)
  ) +
  labs(
    x = "Trial number",
    y = "Proportion correct",
  ) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.1), expand = c(0, 0.001)) +
  scale_color_brewer(palette = "Set1", name = "Word order") +
  scale_fill_brewer(palette = "Set1", name = "Word order") +
  # coord_cartesian(ylim = c(0.2, 1)) +
  theme_classic(base_size = 14) +
  theme(
    legend.position = "right",
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    axis.text = element_text(color = "black"),
    axis.title.x = element_text(margin = margin(t = 5)),
    axis.title.y = element_text(margin = margin(r = 5)),
    axis.title = element_text(margin = margin(b = 5)),
    strip.background = element_rect(fill = "lightgrey", colour = NA),
    strip.text = element_text(face = "bold"),
  )
dev.off()

### plot data and predictions
pdf("predictions-observed.pdf", height = 5, width = 7)
ggplot(data = df_avg) +
  geom_line(aes(x = trial, y = emp, color = order)) +
  geom_line(aes(x = trial, y = pred), color = "black", size = 0.8) +
  facet_wrap(~ order, ncol = 3) +
  scale_x_continuous(
    limits = c(1, 201),
    expand = c(0, 0.1)
  ) +
  labs(
    x = "Trial number",
    y = "Proportion correct",
  ) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.1), expand = c(0, 0.001)) +
  scale_color_brewer(palette = "Set1") +
  # coord_cartesian(ylim = c(0.2, 1)) +
  theme_classic(base_size = 14) +
  theme(
    legend.position = "none",
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    axis.text = element_text(color = "black"),
    axis.title.x = element_text(margin = margin(t = 5)),
    axis.title.y = element_text(margin = margin(r = 5)),
    axis.title = element_text(margin = margin(b = 5)),
    strip.background = element_rect(fill = "lightgrey", colour = NA),
    strip.text = element_text(face = "bold"),
  )
dev.off()