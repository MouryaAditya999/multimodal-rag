"""
Golden evaluation dataset for Week 1 baseline (dense retrieval).
20 verified QA pairs from:
  - B_Tech_AIDS_Curriculum_SNU .pdf
  - mlt mourya.docx
"""

GOLDEN_DATASET = [
    # --- Curriculum PDF (15 pairs) ---
    {
        "question": "What is the full name of the B.Tech program covered in this curriculum?",
        "ground_truth": "B.Tech Artificial Intelligence and Data Science at SNU Chennai, School of Engineering, Department of Computer Science and Engineering.",
        "source_doc": "B_Tech_AIDS_Curriculum_SNU .pdf",
    },
    {
        "question": "What is the course code for Linear Algebra in Semester I?",
        "ground_truth": "MA1001",
        "source_doc": "B_Tech_AIDS_Curriculum_SNU .pdf",
    },
    {
        "question": "Which course teaches Programming in Python in Semester II?",
        "ground_truth": "CS1002 - Programming in Python",
        "source_doc": "B_Tech_AIDS_Curriculum_SNU .pdf",
    },
    {
        "question": "What is the course code for Artificial Intelligence in Semester III?",
        "ground_truth": "CS2007",
        "source_doc": "B_Tech_AIDS_Curriculum_SNU .pdf",
    },
    {
        "question": "Which course covers Machine Learning Techniques in Semester IV?",
        "ground_truth": "CS2012 - Machine Learning Techniques",
        "source_doc": "B_Tech_AIDS_Curriculum_SNU .pdf",
    },
    {
        "question": "What courses are offered in Semester V?",
        "ground_truth": "MA3001 Optimization Techniques, CS3005 Web Technologies, MA3003 Statistical Inference, CS3007 Big Data Analytics, BM3003 Principles of Management, CS3011 Introduction to Robotics, CS3009 Introduction to Digital Signal Processing, plus Big Data Analytics Lab and Web Technologies Lab.",
        "source_doc": "B_Tech_AIDS_Curriculum_SNU .pdf",
    },
    {
        "question": "What is the course code for Natural Language Processing in Semester VI?",
        "ground_truth": "CS3012",
        "source_doc": "B_Tech_AIDS_Curriculum_SNU .pdf",
    },
    {
        "question": "What theory courses are in Semester VII?",
        "ground_truth": "CS4007 Deep Learning, CS4009 Speech Technology, Professional Elective III, Professional Elective IV, and Open Elective III.",
        "source_doc": "B_Tech_AIDS_Curriculum_SNU .pdf",
    },
    {
        "question": "What is the course code for Deep Learning in Semester VII?",
        "ground_truth": "CS4007",
        "source_doc": "B_Tech_AIDS_Curriculum_SNU .pdf",
    },
    {
        "question": "What practical courses are in Semester VII?",
        "ground_truth": "CS4803 Deep Learning Lab and CS4997 Capstone Project I.",
        "source_doc": "B_Tech_AIDS_Curriculum_SNU .pdf",
    },
    {
        "question": "List the Professional Elective III options for Semester VII.",
        "ground_truth": "Semantic Web Technology (CS4687), Media Planning and Strategies (CS4685), Cryptography and Network Security (CS4683), Augmented and Virtual Reality (CS4689), Graphs - Algorithms and Mining (CS4681), and Quantum computing (CS3689).",
        "source_doc": "B_Tech_AIDS_Curriculum_SNU .pdf",
    },
    {
        "question": "List the Professional Elective IV options for Semester VII.",
        "ground_truth": "Reinforcement Learning (CS4679), Ethics for Data Science (CS4677), Data Security and Privacy (CS4675), Microcontrollers for IoT (CS4673), Bioinformatics (CS4671), and GPU Architectures and Programming (CS4669).",
        "source_doc": "B_Tech_AIDS_Curriculum_SNU .pdf",
    },
    {
        "question": "What is the total number of credits for the B.Tech program?",
        "ground_truth": "162 credits",
        "source_doc": "B_Tech_AIDS_Curriculum_SNU .pdf",
    },
    {
        "question": "What courses are in Semester VIII?",
        "ground_truth": "Professional Elective V, Professional Elective VI, and Capstone Project II (CS4996).",
        "source_doc": "B_Tech_AIDS_Curriculum_SNU .pdf",
    },
    {
        "question": "What is the course code for Big Data Analytics in Semester V?",
        "ground_truth": "CS3007",
        "source_doc": "B_Tech_AIDS_Curriculum_SNU .pdf",
    },
    # --- MLT Lab DOCX (5 pairs) ---
    {
        "question": "What is the title of the MLT lab experiment document?",
        "ground_truth": "Decision Tree Classification on Airlines Booking Dataset and Bakery Sales Dataset",
        "source_doc": "mlt mourya.docx",
    },
    {
        "question": "What type of classification problem is the Airlines Booking Dataset?",
        "ground_truth": "Binary classification with 2 classes (Confirmed and Cancelled booking status).",
        "source_doc": "mlt mourya.docx",
    },
    {
        "question": "What is the target variable in the Airlines Booking Dataset?",
        "ground_truth": "Booking Status (Confirmed or Cancelled).",
        "source_doc": "mlt mourya.docx",
    },
    {
        "question": "What criterion is used for the Decision Tree classifier in the Airlines experiment?",
        "ground_truth": "Entropy",
        "source_doc": "mlt mourya.docx",
    },
    {
        "question": "How many input features does the Airlines Booking Dataset have?",
        "ground_truth": "10 input features",
        "source_doc": "mlt mourya.docx",
    },
]


def get_golden_dataset():
    """Return the full golden evaluation dataset."""
    return GOLDEN_DATASET
