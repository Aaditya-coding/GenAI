from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):

    name: str = 'aditya'
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0, lt=10, default=6, description='A decimal value representing the cgpa of the student.')

new_student = {'age':32, 'email':'ABC@outlook.com'}

student = Student(**new_student)

student_dict = dict(student)
student_json = student.model_dump_json()

print(student_dict['age'])
print(student_json)