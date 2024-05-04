from flask import Flask, request, jsonify
import RAG
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('user_input')
    # Process user input and generate bot response
    bot_response = RAG.interact_with_chatbot(user_input)
    return jsonify({'bot_response': bot_response})

def get_user_input():
    subject = input("What is your subject? ")
    topic = input("What is the topic you want to teach? ")
    class_size = int(input("What is your class size? "))
    class_duration = int(input("What is your class duration (in minutes)? "))
    engineering_major = input("What is the engineering major of your students? ")
    mode_of_teaching = input("How do you currently teach the course? (e.g., online/offline) ")
    # course_structure = input("Can you describe the course structure and format? ")
    # course_description = input("Can you provide a brief description of the course content? ")

    additional_info = ""
    more_info = input("Do you have any additional information about your course or students? (y/n) ")
    if more_info.lower() == "y":
        additional_info = input("Please provide any additional information: ")

    user_input = f"subject: {subject}\ntopic: {topic}\nclass_size: {class_size}\nclass_duration: {class_duration}\nengineering_major: {engineering_major}\nmode_of_teaching: {mode_of_teaching}\nadditional_info: {additional_info}\n"
    # user_input = f"I am a professor and I want to teach the topic:{topic},{subject}, to engineering_major {engineering_major}. The class size is {class_size} and class_duration is {class_duration} mins. The class is taken in {mode_of_teaching} mode. Please tell me what Active Learning methods can I use for the same, also give me steps on hot to implement those methods and resources and citations.\nadditional_info: {additional_info}\n"

    return user_input


if __name__ == '__main__':
    app.run(debug=True)
