import json
from typing import Optional

def load_questions(question_file: str, begin: Optional[int], end: Optional[int]):
    questions = []
    with open(question_file, "r") as ques_file:
        for line in ques_file:
            if line:
                questions.append(json.loads(line))
    questions = questions[begin:end]
    return questions