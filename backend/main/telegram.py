from telegram import Bot
import asyncio

your_telegram_bot_token = "7878300541:AAHF2aF3v2cxXkePujiMNsFv3FCPQu0LT5A"

async def send_telegram_message(otp, chat_id):
    """Send OTP to a specific Telegram user."""
    try:
        bot = Bot(token=your_telegram_bot_token)
        message = f"Your OTP is: {otp}"
        await bot.send_message(chat_id=chat_id, text=message)  # ✅ Await async function
        print("✅ OTP sent successfully")
    except Exception as e:
        print(f"❌ Error sending OTP: {e}")

# Run the async function
new_otp = "6645"
chat_id = 6328640036

asyncio.run(send_telegram_message(new_otp, chat_id))  # ✅ This ensures the async function runs properly
