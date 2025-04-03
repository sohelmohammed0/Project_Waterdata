document.getElementById('sendBtn').addEventListener('click', sendMessage);
document.getElementById('userInput').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendMessage();
});

function sendMessage() {
    const userInput = document.getElementById('userInput').value.trim();
    if (!userInput) return;

    appendMessage('You', userInput, 'bg-gray-700 text-green-400 text-right');
    document.getElementById('userInput').value = '';

    fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userInput })
    })
    .then(response => {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let botMessage = '';
        const msgDiv = appendMessage("ARSHU'S Chatbot", '', 'bg-gray-800 text-green-300 text-left');

        function readStream() {
            reader.read().then(({ done, value }) => {
                if (done) return;
                botMessage += decoder.decode(value, { stream: true });
                msgDiv.innerHTML = `<strong>ARSHU'S Chatbot:</strong> ${botMessage}`;
                document.getElementById('chatbox').scrollTop = document.getElementById('chatbox').scrollHeight;
                readStream();
            });
        }
        readStream();
    })
    .catch(error => {
        appendMessage("ARSHU'S Chatbot", 'System error detected!', 'bg-red-900 text-red-400 text-left');
        console.error('Fetch error:', error);
    });
}

function appendMessage(sender, message, className) {
    const chatbox = document.getElementById('chatbox');
    const msgDiv = document.createElement('div');
    msgDiv.className = `p-3 my-2 rounded-lg ${className}`;
    msgDiv.innerHTML = `<strong>${sender}:</strong> ${message}`;
    chatbox.appendChild(msgDiv);
    chatbox.scrollTop = chatbox.scrollHeight;
    return msgDiv;
}