# src/db_ops.py
import sqlite3
import pandas as pd
import os
import random # Needed for shuffling the data

DB_NAME = 'spam_database.db'

def init_db():
    """Initializes the database and creates the messages table."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY,
            message_text TEXT NOT NULL,
            true_label TEXT NOT NULL,
            priority INTEGER DEFAULT 1,
            is_used_for_training BOOLEAN DEFAULT FALSE
        )
    """)
    conn.commit()
    conn.close()

def insert_message(message_text, true_label, priority=1, is_used=False):
    """Inserts a single message into the database (used by the /feedback route)."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute(
        "INSERT INTO messages (message_text, true_label, priority, is_used_for_training) VALUES (?, ?, ?, ?)",
        (message_text, true_label, priority, is_used)
    )
    conn.commit()
    conn.close()

def fetch_training_data():
    """Fetches all data for training (including priority)."""
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT message_text, true_label, priority FROM messages", conn)
    conn.close()
    return df

def update_training_status():
    """Marks all current records as used after a training session."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("UPDATE messages SET is_used_for_training = TRUE")
    conn.commit()
    conn.close()

def get_500_sample_data():
    """Generates exactly 500 samples by repeating the 30 unique messages."""
    
    unique_data = [
        # HAM SAMPLES (15 total)
        ("Confirmed appointment for 3 PM on Tuesday. Don't be late!", "ham", 1),
        ("The new project requirements document has been uploaded to the shared drive.", "ham", 1),
        ("Can we reschedule the client review meeting to Wednesday afternoon?", "ham", 1),
        ("Tickets booked for the concert at 8 PM. Picking you up at 7:00.", "ham", 1),
        ("Reminder: The annual software license renewal is due next month.", "ham", 1),
        ("The invoice for electricity payment is due next week. Please pay by the 10th.", "ham", 1),
        ("Your Amazon order 702-8930 has shipped and will arrive tomorrow.", "ham", 1),
        ("Please submit the weekly report before the end of the day today.", "ham", 1),
        ("I sent the final receipt for the hardware purchase yesterday.", "ham", 1),
        ("Let's meet for lunch at the cafe around noon?", "ham", 1),
        ("Received the confirmation email for the flight booking. Looks good.", "ham", 1),
        ("Did you manage to find that document in the folder we discussed?", "ham", 1),
        ("My train arrives at 6 PM. Can you pick me up from the station?", "ham", 1),
        ("The maintenance fee for the condo is due next month.", "ham", 1),
        ("The team meeting location has been moved to Conference Room C. Please note the change.", "ham", 1),
        
        # SPAM SAMPLES (15 total)
        ("URGENT! Claim your FREE $5,000 prize NOW before it expires!", "spam", 1),
        ("Your acct has been suspended. Verify your bank info immediately at this link.", "spam", 1),
        ("We are liquidating assets! Last chance to invest! Limited time offer.", "spam", 1),
        ("Congrats! You qualify for a FREE gift card. Just reply 'GIFT' to this message.", "spam", 1),
        ("Is your credit score low? We can fix it instantly! Call 800-555-4000.", "spam", 1),
        ("Top Secret opportunity! Get rich quick with our guaranteed scheme.", "spam", 1),
        ("You are selected! Get a brand new smartphone for only $1. Shipping fee applies.", "spam", 1),
        ("Final warning: Your system has a security breach! Download our app immediately.", "spam", 1),
        ("Claim your exclusive 50% discount code by visiting the following website.", "spam", 1),
        ("The deadline for the phase one summary is Monday.", "spam", 1),
        ("Wire transfer pending. To approve, enter your full password here.", "spam", 1),
        ("Congratulations! You have inherited a large sum of money from a distant relative.", "spam", 1),
        ("Stop receiving these emails? Unsubscribe here to stop future notifications.", "spam", 1),
        ("Limited quantity available! Buy now before this special offer runs out!", "spam", 1),
        ("RE: Your outstanding invoice—pay now to avoid late payment fees.", "spam", 1)
    ]

    # Calculate repetitions: 500 / 30 = 16.66. We'll repeat 16 times (480 total)
    REPETITIONS = 16
    
    # 1. Start with 480 samples (16 * 30)
    data_500_samples = unique_data * REPETITIONS
    
    # 2. Add the remaining 20 samples (500 - 480)
    data_500_samples.extend(unique_data[:20])
    
    # Shuffle the final list to mix Ham and Spam samples before insertion
    random.shuffle(data_500_samples)

    return data_500_samples


def insert_bulk_data():
    """Inserts all 500 generated samples directly into the database."""
    data = get_500_sample_data()
    
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    # Insert data (message_text, true_label, priority)
    c.executemany("INSERT INTO messages (message_text, true_label, priority) VALUES (?, ?, ?)", data)
    conn.commit()
    conn.close()
    return len(data)

if __name__ == '__main__':
    # Initialize table
    init_db()
    
    # Check if data exists before inserting to prevent duplicates
    if fetch_training_data().shape[0] < 500:
        count = insert_bulk_data()
        print(f"Database initialized and {count} samples inserted.")
    else:
        print(f"Database initialized. {fetch_training_data().shape[0]} samples already present.")