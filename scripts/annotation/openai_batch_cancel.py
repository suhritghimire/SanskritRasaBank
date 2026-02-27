OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)
batch_id = "batch_699decd18fec8190966401357b3b4e55"

try:
    print(f"Attempting to cancel batch: {batch_id}")
    client.batches.cancel(batch_id)
    print("Cancellation request submitted.")
except Exception as e:
    print(f"Error canceling batch: {e}")
