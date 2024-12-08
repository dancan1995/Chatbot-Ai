// Function to send the user input to the backend API and get the response
function sendMessage() {
    const userInput = document.getElementById('user-input').value;
    if (userInput.trim() === '') return;

    // Add user message to chat box
    appendMessage('You: ' + userInput);

    // Send message to API
    fetch('/api/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: userInput }),
    })
    .then(response => response.json())
    .then(data => {
        // Display bot response in chat box
        appendMessage('Bot: ' + data.message);
    })
    .catch(error => {
        console.error('Error:', error);
    });

    // Clear input field
    document.getElementById('user-input').value = '';
}

// Function to append messages to chat box
function appendMessage(message) {
    const chatBox = document.getElementById('chat-box');
    const messageElement = document.createElement('div');
    messageElement.textContent = message;
    chatBox.appendChild(messageElement);
    chatBox.scrollTop = chatBox.scrollHeight; // Scroll to the bottom
}
