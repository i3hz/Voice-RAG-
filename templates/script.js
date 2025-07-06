const recordButton = document.getElementById('record-button');
const questionInput = document.getElementById('question-input');
const sendButton = document.getElementById('send-button');
const chatContainer = document.getElementById('chat-container');
const audioResponse = document.getElementById('audio-response');
let isRecording = false;

function showTypingIndicator() {
    const indicator = document.createElement('div');
    indicator.className = 'typing-indicator';
    indicator.innerHTML = `
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
    `;
    chatContainer.appendChild(indicator);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    return indicator;
}

recordButton.addEventListener('click', async () => {
    if (!isRecording) {
        try {
            const response = await fetch('/start_recording', { method: 'POST' });
            if (response.ok) {
                isRecording = true;
                recordButton.innerHTML = '<i class="fas fa-stop"></i> Stop';
                recordButton.classList.add('recording');
            }
        } catch (error) {
            console.error('Error starting recording:', error);
        }
    } else {
        try {
            const response = await fetch('/stop_recording', { method: 'POST' });
            const data = await response.json();
            questionInput.value = data.text;

            isRecording = false;
            recordButton.innerHTML = '<i class="fas fa-microphone"></i> Record';
            recordButton.classList.remove('recording');
        } catch (error) {
            console.error('Error stopping recording:', error);
        }
    }
});

async function sendQuestion() {
    const question = questionInput.value.trim();
    if (!question) return;

    appendMessage('question', question);
    questionInput.value = '';

    const typingIndicator = showTypingIndicator();

    try {
        const response = await fetch('/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question: question,
                format: audioResponse.checked ? 'audio' : 'text',
            }),
        });

        typingIndicator.remove();

        const data = await response.json();
        if (data.error) {
            appendMessage('answer', `Error: ${data.error}`);
        } else {
            appendMessage('answer', data.answer);
            if (data.audio) playAudioResponse(data.audio);
        }
    } catch (error) {
        typingIndicator.remove();
        appendMessage('answer', `Error: ${error.message}`);
    }
}

function appendMessage(type, text) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;
    messageDiv.textContent = text;
    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function playAudioResponse(audioData) {
    const audio = new Audio('data:audio/mp3;base64,' + btoa(audioData));
    audio.play();
}

sendButton.addEventListener('click', sendQuestion);
questionInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendQuestion();
});
