
import streamlit as st
import pickle

# Load the model and vectorizer
model = pickle.load(open('spam123.pk1', 'rb'))
cv = pickle.load(open('vec123.pk1', 'rb'))

def main():
    st.title("Email Spam Classification Application")
    st.write("This is a Machine Learning application to classify emails as spam or ham.")
    st.header("Classification")
    
    user_input = st.text_area("Enter an email to classify", height=150)
    
    if st.button("Classify"):
        if user_input:
            data = [user_input]
            st.write("Processing input...")
            vec = cv.transform(data).toarray()
            result = model.predict(vec)
            
            if result[0] == 0:
                st.success("This is Not A Spam Email")
            else:
                st.error("This is A Spam Email")
        else:
            st.write("Please enter an email to classify.")

# Run the main function
main()
    