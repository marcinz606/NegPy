# 🚀 How to Run This App (The Easy Way)

Welcome! You don't need to be a programmer to run this project. We use a tool called **Docker**, which acts like a "container" for the app so it works on your computer exactly like it works on mine—no complicated setup required.

---

## 1. Install Docker Desktop
Think of Docker as the "player" and this project as the "game." You need the player installed first.

* **For Windows:** [Download Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/)
    * *Note:* If it asks about **WSL 2**, say **Yes**. You may need to restart your computer once.
* **For Mac:** [Download Docker Desktop for Mac](https://www.docker.com/products/docker-desktop/)
    * *Note:* Choose **Apple Chip** for Apple Silicon Macs, or **Intel Chip** for older ones.

**How to know it's working:** Open the Docker Desktop app. If you see a green status bar at the bottom, you're ready!

---

## 2. Download This Project
You don't need "Git." Just get the files:
1.  Scroll to the top of this GitHub page.
2.  Click the green **<> Code** button.
3.  Click **Download ZIP**.
4.  Unzip the folder onto your **Desktop** or into your **Documents**.

---

## 3. Launch the App
This is the only part where we use the "Terminal" (the text window), but it's just one command!

1.  **Open your Terminal:**
    * **Windows:** Press the `Start` key, type `PowerShell`, and hit Enter.
    * **Mac:** Press `Cmd + Space`, type `Terminal`, and hit Enter.
2.  **Go to the folder:**
    * Type `cd` and then a **space**.
    * **Drag the unzipped folder** from your desktop directly into that window. It will automatically paste the folder's location.
    * Hit **Enter**.
3.  **Start it up:**
    * Copy and paste this command and hit Enter:
      ```bash
      docker compose up -d
      ```
4.  **Wait a minute:** The first time you do this, it downloads the "ingredients" for the app. When the text stops moving, you are done!

---

## 4. How to Use the App
1.  Open the **Docker Desktop** app.
2.  Click on the **Containers** tab on the left.
3.  Find this project in the list. 
4.  Look for the blue link under the **Port(s)** column (it usually looks like `8080:80`).
5.  **Click that link!** Your web browser will open, and the app will be running.

---

## 🔄 How to Update (When a new version is out)
If I release an update, follow these steps to "refresh" your app:

1.  **Download the new ZIP** (just like in Step 2) and replace your old folder.
2.  Open your **Terminal** or **PowerShell** again.
3.  `cd` into the folder (just like in Step 3).
4.  Run this command:
    ```bash
    docker compose up -d --build
    ```
    *This tells Docker: "Look for changes and rebuild the app with the new stuff."*

---

## 🛑 How to Turn It Off
1.  In **Docker Desktop**, go to **Containers**.
2.  Click the square **Stop** button next to the project.
3.  To start it again later, just click the triangle **Start** button!

---

**Troubleshooting for Windows:**
If you get an error about "WSL," open PowerShell, type `wsl --update`, and restart Docker.
