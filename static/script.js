document.addEventListener('DOMContentLoaded', () => {
    const messageForm = document.getElementById('message-form');
    const userInput = document.getElementById('user-input');
    const chatBox = document.getElementById('chat-box');
    const resetButton = document.getElementById('reset-button');
    const initialChatHTML = chatBox.innerHTML;

    messageForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const messageText = userInput.value.trim();
        if (messageText === '') return;

        addMessageToUI(messageText, 'user');
        userInput.value = '';

        const requestBody = { type: 'text', message: messageText };
        sendRequest(requestBody);
    });

    resetButton.addEventListener('click', async () => {
        try {
            await fetch('/reset', { method: 'POST' });
        } catch (error) {
            console.error('Failed to reset backend state:', error);
        }
        chatBox.innerHTML = initialChatHTML;
        userInput.focus();
    });

    async function sendRequest(body) {
        showTypingIndicator();
        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
            });

            removeTypingIndicator();
            if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);

            const data = await response.json();
            handleBotResponse(data);

        } catch (error) {
            console.error('Error:', error);
            removeTypingIndicator();
            addMessageToUI('Sorry, something went wrong. Please try again.', 'assistant');
        }
    }

    function handleBotResponse(data) {
        if (data.type === 'multiple_problems_selection') {
            addMessageToUI(data.content, 'assistant');
            createProblemSelectionButtons(data.problems);
        } else {
            addMessageToUI(data.content, 'assistant');
        }
    }

    function createProblemSelectionButtons(problems) {
        const container = document.createElement('div');
        container.className = 'problem-selection-container';

        problems.forEach(problem => {
            const button = document.createElement('button');
            button.className = 'problem-choice-button';
            button.textContent = problem.current_user_problem || problem.problem_text || 'Problem';
            button.dataset.problemId = problem.problem_id;

            button.addEventListener('click', () => {
                // Disable all buttons after one is clicked
                container.querySelectorAll('.problem-choice-button').forEach(btn => {
                    btn.disabled = true;
                });

                const userChoiceText = `I'll select: "${button.textContent}"`;
                addMessageToUI(userChoiceText, 'user');

                const requestBody = {
                    type: 'problem_selection',
                    selected_problem_id: button.dataset.problemId
                };
                sendRequest(requestBody);
            });
            container.appendChild(button);
        });
        chatBox.appendChild(container);
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    function addMessageToUI(text, sender) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message', `${sender}-message`);
        messageElement.innerHTML = `<p>${text}</p>`;
        chatBox.appendChild(messageElement);
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    function showTypingIndicator() {
        if (document.getElementById('typing-indicator')) return;
        const indicator = document.createElement('div');
        indicator.id = 'typing-indicator';
        indicator.className = 'message assistant-message typing-indicator';
        indicator.innerHTML = '<span></span><span></span><span></span>';
        chatBox.appendChild(indicator);
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    function removeTypingIndicator() {
        const indicator = document.getElementById('typing-indicator');
        if (indicator) indicator.remove();
    }
});