# Initial Task Solutions

## Task 1: Converting CSV File to Database

The solution involves converting the CSV file into a database and decomposing the question into subqueries (query decomposition) that can be understood by the system using LLM. The steps are as follows:

1. **Split the question into subqueries:**

   - Decompose the main query into smaller, manageable subqueries.

2. **Loop over the subqueries and query the database file:**

   - Iterate through each subquery and execute it against the database.

3. **Retrieve the results and prompt the LLM to present them in a presentable form:**

   - Collect the results and use LLM to format and present them.

4. **Ensure the system tolerates query errors and makes multiple attempts if results are not found:**
   - Implement error handling to manage failed queries and retry if necessary.

**Tools used:**

- LangChain
- GPT-3.5 Turbo or GPT-4 (both work fine, but the former is more cost-effective)

to run the code `cd` to the `submissionFolder`
create your own .env file and add `API_KEY=YOUR_SECRET_KEY`

then write

```
py peopleDetection.py
```

## Task 2: Counting People in Each Frame

I used YOLOv8x to count the number of people in each frame, excluding the cashier area.

### Steps:

1. **Load the video frames:**

   - Extract frames from the video.

2. **Apply YOLOv8x model:**

   - Use the YOLOv8x model to detect and count the number of people in each frame.

3. **Exclude the cashier area:**
   - Implement logic to exclude detections in the cashier area.

### Potential Enhancements

#### Task 1

- Reducing cost
- Use similarity embeddings to find similar queries to the user's question
- Use query structuring to extract the variables you want to query in general queries

#### Task 2

- Track the objects
- Use the summation of confidence between frames of the tracked objects
- Use a fine-tuned model
- Try different models
