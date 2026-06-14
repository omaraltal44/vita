# VITAPRO Original Frontend + FastAPI Backend

This package keeps the original VitaPro frontend design from the provided ZIP and integrates it with the FastAPI/MySQL/AI backend.

## Backend

```powershell
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

## MySQL setup

For XAMPP PowerShell:

```powershell
Get-Content .\backend\mysql_setup.sql | C:\xampp\mysql\bin\mysql.exe -u root
```

Then confirm:

```powershell
C:\xampp\mysql\bin\mysql.exe -u vitapro_user -pCHANGE_PASSWORD -e "SHOW DATABASES;"
```

## Frontend

```powershell
cd frontend
python -m http.server 5500
```

Open:

```text
http://127.0.0.1:5500
```

The original login/register page now calls `/auth/register` and `/auth/login`.
The existing disease forms now call `/predict/diabetes`, `/predict/hypertension`, `/predict/heart`, and `/predict/obesity`.
