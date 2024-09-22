# streamlit_app.py
import streamlit as st
import requests
import json

# Set the FastAPI backend URL
BACKEND_URL = "http://localhost:8000"

st.title("NCERT Q&A Helper")

# Input box for user question
question = st.text_input("Enter your question:")

if st.button("Ask"):
    if question:
        # Send request to FastAPI backend
        response = requests.post(f"{BACKEND_URL}/ask", json={"question": question})
        
        if response.status_code == 200:
            data = response.json()
            
            # Display the answer
            st.subheader("Answer:")
            st.write(data["answer"])
            
            # Display retrieved documents
            st.subheader("Retrieved Documents:")
            for doc in data["retrieved_documents"]:
                with st.expander(f"Document (Page: {doc['page']})"):
                    st.write(f"Link: {doc['link']}")
                    st.write("Snippet:")
                    st.write(doc['snippet'])
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
    else:
        st.warning("Please enter a question.")

# Add some instructions or information about the system
st.sidebar.header("About")
st.sidebar.info("This is a NCERT Q&A Helper. Enter your question in the text box and click 'Ask' to get an answer based on the available documents.")