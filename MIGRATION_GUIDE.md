# 🛠️ QueryMate Database Migration Guide (Alembic)

We now use **Alembic** to manage database changes. This allows us to update the database schema (add columns, change tables) without deleting existing data.

---

## 🚀 How to Apply Changes (First Time / New Updates)

Whenever you pull new code that includes database changes, run this command in the `backend/` folder:

```bash
cd backend
alembic upgrade head
```
*This ensures your local database matches the latest code.*

---

## 🛠️ How to Update the Database (For Developers)

If you modify `models.py` (e.g., add a new column like `email` to the `User` table), follow these steps:

### 1. Generate a Migration Script
Alembic will compare your `models.py` with your actual database and create a "version" file.
```bash
cd backend
alembic revision --autogenerate -m "Added email column to user"
```
*A new file will appear in `backend/migrations/versions/`.*

### 2. Review the Script (Optional)
Open the newly created file to make sure it contains the changes you intended (e.g., `op.add_column(...)`).

### 3. Apply the Migration
Actually update your database file (`chat_app.db`).
```bash
alembic upgrade head
```

---

## 📜 Standard Commands

| Action | Command |
| :--- | :--- |
| **Apply all updates** | `alembic upgrade head` |
| **Undo last update** | `alembic downgrade -1` |
| **View history** | `alembic history --verbose` |
| **Show current version** | `alembic current` |

---

## 💡 Troubleshooting
If you see an error like `'alembic' is not recognized`, ensure your virtual environment is active and run:
```bash
pip install alembic
```
