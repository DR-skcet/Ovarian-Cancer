from fpdf import FPDF
import os

# Sample data (replace with actual data)
patient_data = {
    'Patient_ID': 1,
    'Age': 45,
    'Genetic_Mutation': 'ABC123',
    'Medical_History': 'No significant medical history',
    'Symptoms': 'Abdominal pain, bloating, fatigue',
    'Family_History': 'No family history of ovarian cancer',
    'Ethnicity': 'Caucasian',
    'BMI': '23.8',
    'Menopausal_Status': 'Postmenopausal',
    'Pregnancy_History': '2 pregnancies, 2 live births',
    'Lifestyle_Factors': 'Non-smoker, moderate alcohol consumption',
    'Dietary_Habits': 'Balanced diet with regular intake of fruits and vegetables',
    'Physical_Activity': 'Regular exercise regimen, 30 minutes/day',
    'Occupation': 'Office job, sedentary lifestyle',
    'Stress_Level': 'Moderate',
    'Sleep_Pattern': '7-8 hours of sleep per night',
    'Medication_Use': 'No regular medication use',
    'Allergies': 'No known allergies',
    'Menstrual_History': 'Regular menstrual cycles, menarche at age 12',
    'Hormone_Therapy': 'No hormone therapy',
    'Contraceptive_Use': 'No current contraceptive use',
    'Surgical_History': 'Appendectomy at age 20',
    'Screening_History': 'Regular mammograms and Pap smears',
    'Other_Diseases': 'None',
    'Current_Medications': 'Vitamin supplements',
    'Alcohol_Use': 'Moderate alcohol consumption, socially',
    'Tobacco_Use': 'Non-smoker',
    'Drug_Use': 'No recreational drug use',
    'Environmental_Exposure': 'No known exposure to carcinogens',
    'Family_History_of_Cancer': 'No family history of ovarian cancer, breast cancer in maternal aunt',
    'Social_History': 'Stable social support, active social life',
    'Psychological_Status': 'No known mental health disorders',
    'Quality_of_Life': 'Good, active and independent lifestyle',
    # Other relevant patient data
}

predicted_subtype = 'Subtype B'  # Replace with actual predicted subtype
outlier_detection = 'Normal'  # Replace with actual outlier detection result
stage_detection = 'Stage 2'  # Replace with actual stage detection result

# Create PDF document
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()
pdf.set_font("Times",size=14)  # Reduced font size to 14

# Add title to the PDF
pdf.cell(200, 10, txt="OVARIAN CANCER SUBTYPE CLASSIFICATION AND OUTLIER DETECTION REPORT", ln=True, align='C')
pdf.ln(10)

# Add patient information to the report
pdf.cell(200, 10, txt="Patient Information:", ln=True)
pdf.ln(5)
for key, value in patient_data.items():
    pdf.cell(0, 10, f"{key}: {value}", ln=True)
pdf.ln(10)

# Add detailed results section
pdf.cell(200, 10, txt="Detailed Results:", ln=True)
pdf.ln(5)

# Add predicted subtype details
pdf.cell(200, 10, txt="Predicted Subtype Details:", ln=True)
pdf.ln(5)
subtype_details = (
    "The model identified several features in the patient's genetic data that align with Subtype B of ovarian cancer, "
    "including mutations in specific genes and expression levels of certain proteins. "
    "These features collectively suggest a high likelihood of Subtype B."
)
pdf.multi_cell(0, 10, txt=subtype_details)
pdf.ln(5)

# Add outlier detection details
pdf.cell(200, 10, txt="Outlier Detection Details:", ln=True)
pdf.ln(5)
outlier_details = (
    "The model analyzed the patient's genetic data for outliers and determined that the genetic data falls within the normal range. "
    "This suggests a typical genetic profile for the patient."
)
pdf.multi_cell(0, 10, txt=outlier_details)
pdf.ln(5)

# Add stage detection details
pdf.cell(200, 10, txt="Stage Detection Details:", ln=True)
pdf.ln(5)
stage_details = (
    "Based on the analysis, the model predicted the patient to be at Stage 2 of ovarian cancer, "
    "indicating a moderate level of cancer progression. "
    "This information is crucial for determining the appropriate treatment plan."
)
pdf.multi_cell(0, 10, txt=stage_details)
pdf.ln(10)

# Add treatment recommendations section
pdf.cell(200, 10, txt="Treatment Recommendations:", ln=True)
pdf.ln(5)
treatment_recommendations = (
    "Based on the subtype prediction and stage detection, "
    "the following treatment recommendations are suggested: [Add your treatment recommendations here]."
)
pdf.multi_cell(0, 10, txt=treatment_recommendations)
pdf.ln(10)

# Add conclusion section
pdf.cell(200, 10, txt="Conclusion:", ln=True)
pdf.ln(5)
conclusion = (
    "This report provides a detailed analysis of the patient's ovarian cancer diagnosis, "
    "including subtype prediction, outlier detection, and stage detection. "
    "The findings are based on advanced machine learning techniques and are intended to guide personalized treatment planning."
)
pdf.multi_cell(0, 10, txt=conclusion)

pdf.add_page()
pdf.cell(100, 10, txt="Histopathological Image 1:", ln=True)
pdf.ln(5)
y_before_image1 = pdf.get_y()  # Get current y position
pdf.image("histopathological_image1.jpg", x=10, y=y_before_image1, w=180)
y_after_image1 = pdf.get_y()  # Get y position after adding image
pdf.set_y(max(y_before_image1, y_after_image1))  # Set y position to the maximum of before and after image positions
pdf.ln(5)

# Add second histopathological image to the report
pdf.add_page()
pdf.ln(5)
y_before_image2 = pdf.get_y()  # Get current y position
pdf.image("histopathological_image2.jpg", x=10, y=y_before_image2, w=180)
y_after_image2 = pdf.get_y()  # Get y position after adding image
pdf.set_y(max(y_before_image2, y_after_image2))  # Set y position to the maximum of before and after image positions
pdf.ln(5)


# Save the PDF
pdf_file_path = "Ovarian_Cancer_Report.pdf"
pdf.output(pdf_file_path)

# Open the PDF file
os.system(pdf_file_path)

print("Report generated successfully.")
