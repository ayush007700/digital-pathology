from src.rag.report.report_generator import ReportGenerator

generator = ReportGenerator()

report = generator.generate(

    prediction="Tumor",

    confidence=98.6,

    llm_answer="""
The retrieved literature suggests that HER2-positive breast cancer
is associated with aggressive tumor behavior and targeted therapy.
""",

    model_name="ResNet18",

)

print(report)