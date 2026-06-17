import pandas as pd
import random

data = []

# Generate 500 records for each class
for performance in ["Poor", "Average", "Good", "Excellent"]:

    for _ in range(500):

        if performance == "Poor":

            attendance = random.randint(40, 60)
            study_hours = round(random.uniform(1, 3), 1)
            assignment_score = random.randint(35, 55)
            previous_test_score = random.randint(30, 55)
            sleep_hours = round(random.uniform(4, 6), 1)
            participation = "Low"
            internet_access = random.choice(["Yes", "No"])
            extra_classes = "No"
            parent_education = random.choice(["School", "Graduate"])
            screen_time = round(random.uniform(6, 10), 1)
            projects_completed = random.randint(0, 2)
            stress_level = "High"

        elif performance == "Average":

            attendance = random.randint(55, 75)
            study_hours = round(random.uniform(2, 5), 1)
            assignment_score = random.randint(50, 70)
            previous_test_score = random.randint(50, 70)
            sleep_hours = round(random.uniform(5, 7), 1)
            participation = random.choice(["Low", "Medium"])
            internet_access = "Yes"
            extra_classes = random.choice(["Yes", "No"])
            parent_education = random.choice(["School", "Graduate"])
            screen_time = round(random.uniform(4, 8), 1)
            projects_completed = random.randint(2, 5)
            stress_level = random.choice(["Medium", "High"])

        elif performance == "Good":

            attendance = random.randint(70, 90)
            study_hours = round(random.uniform(4, 7), 1)
            assignment_score = random.randint(70, 85)
            previous_test_score = random.randint(70, 85)
            sleep_hours = round(random.uniform(6, 8), 1)
            participation = random.choice(["Medium", "High"])
            internet_access = "Yes"
            extra_classes = "Yes"
            parent_education = random.choice(
                ["Graduate", "Postgraduate"]
            )
            screen_time = round(random.uniform(2, 6), 1)
            projects_completed = random.randint(4, 8)
            stress_level = random.choice(["Low", "Medium"])

        else:  # Excellent

            attendance = random.randint(85, 100)
            study_hours = round(random.uniform(6, 10), 1)
            assignment_score = random.randint(85, 100)
            previous_test_score = random.randint(85, 100)
            sleep_hours = round(random.uniform(7, 9), 1)
            participation = "High"
            internet_access = "Yes"
            extra_classes = "Yes"
            parent_education = "Postgraduate"
            screen_time = round(random.uniform(1, 4), 1)
            projects_completed = random.randint(7, 10)
            stress_level = "Low"

        data.append([
            attendance,
            study_hours,
            assignment_score,
            previous_test_score,
            sleep_hours,
            participation,
            internet_access,
            extra_classes,
            parent_education,
            screen_time,
            projects_completed,
            stress_level,
            performance
        ])

columns = [
    "Attendance",
    "Study_Hours",
    "Assignment_Score",
    "Previous_Test_Score",
    "Sleep_Hours",
    "Participation_Level",
    "Internet_Access",
    "Extra_Classes",
    "Parent_Education",
    "Screen_Time",
    "Projects_Completed",
    "Stress_Level",
    "Performance"
]

df = pd.DataFrame(data, columns=columns)

# Shuffle rows
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df.to_csv(
    "data/student_performance.csv",
    index=False
)

print("Dataset Created Successfully")
print()
print(df["Performance"].value_counts())
print()
print("Dataset Shape:", df.shape)