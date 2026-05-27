import streamlit as st
import pandas as pd
import json
import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
load_dotenv()
#Model Setup
llm = ChatOllama(
    model="gpt-oss:20b-cloud",
    base_url=os.getenv("OLLAMA_BASE_URL"),
    headers={
        "Authorization": f"Bearer {os.getenv('OLLAMA_API_KEY')}"
    }
)
#Page Config
st.set_page_config(
    page_title="AI Email Task Manager",
    page_icon="📧",
    layout="wide"
)

st.title("AI Email Task Manager")

st.markdown("""
This AI system:
Extracts tasks from emails  
Assigns priorities  
Detects deadlines  
Removes duplicate tasks  
Maintains cumulative dashboard  
Sorts by priority and deadline  
""")
if "all_tasks" not in st.session_state:
    st.session_state.all_tasks = []
if "email_count" not in st.session_state:
    st.session_state.email_count = 0

#Email Input
email_text = st.text_area(
    "Paste Email Content",
    height=300,
    placeholder="Paste your email here..."
)
#Analyze Button
if st.button("Analyze Email"):
    if email_text.strip() == "":
        st.warning("Please enter email content.")
    else:
        st.session_state.email_count += 1
        with st.spinner("Analyzing Email..."):
            #Prompt
            prompt = f"""
            You are an intelligent email assistant.
            The email may contain:
            - greetings
            - casual conversations
            - signatures
            - appreciation messages
            - unrelated text

            Your task is to identify ONLY actionable tasks.

            For each task extract:
            1. Task
            2. Priority (High, Medium, Low)
            3. Deadline

            Also generate a short summary.

            Return ONLY valid JSON.

            JSON format:

            {{
              "summary": "short summary",
              "tasks": [
                {{
                  "task": "task name",
                  "priority": "High",
                  "deadline": "deadline"
                }}
              ]
            }}

            Email:
            {email_text}
            """
            try:
                #LLM API CALL
                response = llm.invoke(prompt)
                result = response.content.strip()
                #LLM Response
                if result.startswith("```json"):
                    result = result.replace("```json", "")
                    result = result.replace("```", "").strip()
                #Parse JSON
                data = json.loads(result)
                summary = data["summary"]
                tasks = data["tasks"]
                #Display Summary 
                st.subheader(f"📌 Email {st.session_state.email_count} Summary")

                st.info(summary)
                #Process Tasks
                for new_task in tasks:
                    new_task_name = new_task["task"].strip().lower()
                    duplicate_found = False
                    for existing_task in st.session_state.all_tasks:
                        existing_name = existing_task["task"].strip().lower()
                        #Duplicate Tasks Checking 
                        if existing_name == new_task_name:
                            duplicate_found = True
                            # Priority Ranking
                            priority_rank = {
                                "High": 3,
                                "Medium": 2,
                                "Low": 1
                            }
                            #Update Priority 
                            if priority_rank.get(new_task["priority"], 0) > priority_rank.get(existing_task["priority"], 0):
                                existing_task["priority"] = new_task["priority"]
                            #Update Deadline
                            existing_task["deadline"] = new_task["deadline"]
                            break

                    #Add new task
                    if not duplicate_found:
                        new_task["email_source"] = f"Email {st.session_state.email_count}"
                        st.session_state.all_tasks.append(new_task)

                #Creating the DataFrame
                df = pd.DataFrame(st.session_state.all_tasks)

                #Priority Sorting
                priority_order = {
                    "High": 1,
                    "Medium": 2,
                    "Low": 3
                }
                #Deadline Sorting
                deadline_order = {
                    "Today": 1,
                    "Tomorrow": 2,
                    "Monday": 3,
                    "Tuesday": 4,
                    "Wednesday": 5,
                    "Thursday": 6,
                    "Friday": 7,
                    "Saturday": 8,
                    "Sunday": 9,
                    "Next Week": 10,
                    "Later": 11,
                    "No Deadline": 12
                }

                #Sorting Columns
                df["priority_rank"] = df["priority"].map(priority_order)

                df["deadline_rank"] = df["deadline"].map(
                    lambda x: deadline_order.get(str(x).title(), 100)
                )

                # Sort the DataFrame
                df = df.sort_values(
                    by=["priority_rank", "deadline_rank"]
                )
                #Remove Helper Columns 
                df = df.drop(
                    columns=["priority_rank", "deadline_rank"]
                )
                #Metrics
                high_count = len(df[df["priority"] == "High"])
                medium_count = len(df[df["priority"] == "Medium"])
                low_count = len(df[df["priority"] == "Low"])
                col1, col2, col3 = st.columns(3)
                col1.metric("🔴 High Priority", high_count)
                col2.metric("🟡 Medium Priority", medium_count)
                col3.metric("🟢 Low Priority", low_count)
                #Color Function
                def highlight_priority(val):
                    if val == "High":
                        return "background-color: red; color: white"

                    elif val == "Medium":
                        return "background-color: orange; color: white"

                    elif val == "Low":
                        return "background-color: green; color: white"

                    return ""
                #Display Table
                st.subheader("📋 Cumulative Task Dashboard")
                styled_df = df.style.map(
                    highlight_priority,
                    subset=["priority"]
                )
                st.dataframe(
                    styled_df,
                    use_container_width=True
                )
                #Download CSV 
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇ Download Tasks CSV",
                    data=csv,
                    file_name="all_tasks.csv",
                    mime="text/csv"
                )
            except json.JSONDecodeError:
                st.error("Invalid JSON returned from model.")
                st.text(result)
            except Exception as e:
                st.error(f"Error: {e}")