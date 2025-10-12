# InsightInvest
InsightInvest is a smart chatbot that helps you analyze any publicly traded company with just a name or ticker symbol. Whether you're a student, beginner investor, or finance enthusiast, InsightInvest gives you a deep, easy-to-understand investment outlook report in seconds.


investchat folder is the frontend next.js app file
backend contains the backend code main.py and news_scrapping.py

to run the project
first cd into backend folder then run -> uvicorn main:app --reload
this will start ur backend server

now in another terminal
cd to investchat folder(in ur terminal u should see InsightInvest\investchat> )
now run -> npm run dev

this will run the ui part in ur browser

to change anything in components change the file called Chat.tsx(this is the ui)
and in app/api/analyze(this is the path of folders) go to route.ts file
this route.ts file is the connector between backend and frontend