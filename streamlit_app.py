import streamlit as st
import requests

#streamlit
st.title("GPT-2 Chatbot")

st.write("This is a chatbot powered by a fine-tuned GPT-2 model. Ask any question and get a response!")

user_input = st.text_input("You: ","")

if st.button("Send"):
  if user_input:
    #send the user input to the FastAPI backend
    response = requests.post('https://dpatel9923-face-expression-prediction.hf.space/prediction',json={'prompt':user_input})
    response_json = response.json()
    bot_response = response_json['response']
    st.text_area("Bot:", bot_response, height = 200)

  else:
    st.write("Please enter a question to get a response.")

if __name__ == '__main__':
  st.run()
