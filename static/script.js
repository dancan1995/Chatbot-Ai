document.getElementById("send-btn").addEventListener("click", function() {
    const userInput = document.getElementById("user-input").value;
    if (userInput.trim() === "") return;

    // Display user message
    const userMessage = document.createElement("div");
    userMessage.classList.add("user-message");
    userMessage.textContent = `You: ${userInput}`;
    document.getElementById("chat-box").appendChild(userMessage);

    // Send request to the Flask backend
    fetch("/get-response", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ message: userInput })
    })
    .then(response => response.json())
    .then(data => {
        // Display GPT response
        const gptMessage = document.createElement("div");
        gptMessage.classList.add("gpt-message");
        gptMessage.textContent = `Bot: ${data.message}`;
        document.getElementById("chat-box").appendChild(gptMessage);
    })
    .catch(error => {
        console.error("Error:", error);
    });

    // Clear the input
    document.getElementById("user-input").value = "";
});
