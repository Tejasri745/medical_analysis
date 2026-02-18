from blur_detection import check_blur
from exposure_analysis import analyze_exposure


def final_quality_check(image_path):

    # -----------------------------
    # 1️⃣ Blur Detection
    # -----------------------------
    blur_score, blur_status = check_blur(image_path, threshold=200)

    # -----------------------------
    # 2️⃣ Exposure Detection
    # -----------------------------
    mean_intensity, dark_ratio, bright_ratio, exposure_status = analyze_exposure(image_path)

    # -----------------------------
    # 3️⃣ Quality Score Calculation
    # -----------------------------
    quality_score = 100  # Start with perfect score

    # Blur penalty
    if blur_status == "BLURRY":
        quality_score -= 40

    # Exposure penalty
    if exposure_status == "UNDEREXPOSED":
        quality_score -= 30
    elif exposure_status == "OVEREXPOSED":
        quality_score -= 30

    # Prevent negative score
    quality_score = max(quality_score, 0)

    # -----------------------------
    # 4️⃣ Final Decision Logic
    # -----------------------------
    if quality_score >= 80:
        final_status = "ACCEPTED - VALID FOR DIAGNOSIS"
    else:
        final_status = "REJECTED - QUALITY ISSUES"

    # -----------------------------
    # 5️⃣ Return Full Report
    # -----------------------------
    return {
        "Blur Score": round(blur_score, 2),
        "Blur Status": blur_status,
        "Mean Intensity": round(mean_intensity, 2),
        "Dark Pixel Ratio": round(dark_ratio, 3),
        "Bright Pixel Ratio": round(bright_ratio, 3),
        "Exposure Status": exposure_status,
        "Quality Score (%)": quality_score,
        "Final Decision": final_status
    }


# ------------------------------------------------
# 🔹 Run Directly (Testing Mode)
# ------------------------------------------------
if __name__ == "__main__":

    image_path = "C:/Users/TEJASRI/Desktop/medical_image_validator/datasets/COVID/COVID_3.png"


    report = final_quality_check(image_path)

    print("\n========== IMAGE QUALITY REPORT ==========\n")

    for key, value in report.items():
        print(f"{key}: {value}")

    print("\n==========================================\n")
