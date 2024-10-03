// Add an event listener to the form submission for handling chat input
document.getElementById('text-form').addEventListener('submit', function(event) {
    event.preventDefault(); // Prevent form from refreshing the page
    const inputBox = document.getElementById('message-input'); // Get the input box element
    const message = inputBox.value.trim(); // Get the message, removing leading/trailing spaces
    if (message) {
        addMessageToChat('You', message, 'user-message'); // Add the user's message to the chat
        inputBox.value = ''; // Clear the input field
        inputBox.focus(); // Set focus back to the input field

        // Add a placeholder in the chat for the bot's response
        const botMessageElement = addMessageToChat('Bot', '', 'bot-message');

        // Send the user message to the backend using the Fetch API
        fetch('/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question: message }) // Send the user's message in the request body
        })
        .then(response => {
            const reader = response.body.getReader(); // Get a reader for the streamed response
            const decoder = new TextDecoder('utf-8'); // Decode the response as UTF-8
            let botMessage = ''; // Variable to store the bot's message as it streams

            // Function to process the streamed response
            function read() {
                reader.read().then(({ done, value }) => {
                    if (done) {
                        // When streaming is complete, update the bot message
                        processFinalBotMessage(botMessageElement, botMessage);
                        scrollToBottom(); // Ensure the chat scrolls to the bottom
                        return;
                    }
                    const chunk = decoder.decode(value); // Decode the chunk
                    const lines = chunk.split('\n\n'); // Split the chunk by line breaks
                    lines.forEach(line => {
                        if (line.startsWith('data: ')) {
                            const token = line.replace('data: ', ''); // Remove the 'data: ' prefix
                            botMessage += token; // Append the token to the bot's message
                            updateBotMessage(botMessageElement, botMessage); // Update the chat with the bot's message
                            scrollToBottom(); // Scroll down after updating
                        }
                    });
                    read(); // Read the next chunk
                });
            }
            read();
        })
        .catch(error => {
            updateBotMessage(botMessageElement, 'Error getting response.'); // Show error if request fails
            scrollToBottom(); // Ensure scroll after error
        });
    }
});

// Function to add a new message to the chat
function addMessageToChat(sender, message, type) {
    const chatBox = document.getElementById('chat-content'); // Get the chat content container
    const messageWrapper = document.createElement('div'); // Create a div to wrap the message
    messageWrapper.classList.add('message'); // Add class for message styling

    const messageElement = document.createElement('div'); // Create the message element
    messageElement.innerHTML = `${message}`; // Set the message content
    messageElement.classList.add(type); // Apply specific styling based on the message type (user or bot)

    messageWrapper.appendChild(messageElement); // Add the message to the wrapper
    chatBox.appendChild(messageWrapper); // Add the wrapper to the chat content
    scrollToBottom(); // Scroll to the bottom of the chat

    return messageElement; // Return the message element for further processing
}

// Function to process the final message from the bot
function processFinalBotMessage(messageElement, message) {
    if (message.includes('Sources:')) {
        // If the bot message contains sources, split the message into parts
        const parts = message.split('Sources:');
        messageElement.innerHTML = parts[0]; // Show the main message

        const sourcesElement = document.createElement('div'); // Create an element for the sources
        sourcesElement.textContent = 'Sources: ' + parts[1]; // Set the sources content
        sourcesElement.classList.add('sources'); // Add styling for the sources
        messageElement.appendChild(sourcesElement); // Append the sources to the message
    } else {
        messageElement.innerHTML = message; // Show the full message if no sources
    }

    addCopyButton(messageElement); // Add the copy button to the message
    scrollToBottom(); // Scroll to the bottom after the final message is processed
}

// Function to update the bot's message as it streams in
function updateBotMessage(messageElement, message) {
    messageElement.innerHTML = message; // Update the message content
    setTimeout(scrollToBottom, 100); // Scroll to the bottom after a short delay
}

// Function to add a copy button to the bot's message
function addCopyButton(messageElement) {
    const copyButton = document.createElement('button'); // Create the copy button
    copyButton.classList.add('copy-icon'); // Add a class for styling the button
    copyButton.innerHTML = '<img src="/static/copy-icon.jpg" alt="Copy">'; // Set the button image
    copyButton.addEventListener('click', function() {
        let textToCopy = messageElement.innerText; // Get the message text
        const sourcesElement = messageElement.parentElement.querySelector('.sources'); // Check for sources
        if (sourcesElement) {
            textToCopy += '\n' + sourcesElement.textContent; // Include sources in the text to copy
        }
        copyToClipboard(textToCopy); // Copy the message to the clipboard
    });
    messageElement.appendChild(copyButton); // Add the copy button to the message
    scrollToBottom(); // Scroll to the bottom after the button is added
}

// Modified scrollToBottom function to ensure the chat always shows the latest message
function scrollToBottom() {
    const chatBox = document.getElementById('chat-content'); // Get the chat content container
    setTimeout(() => {
        chatBox.scrollTop = chatBox.scrollHeight; // Scroll to the bottom
    }, 100); // Delay to ensure the DOM is updated before scrolling
}

// Function to copy text to the clipboard
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(function() {
        alert('Copied to clipboard!'); // Show an alert when copying is successful
    }, function(err) {
        alert('Could not copy text: ', err); // Show an error if copying fails
    });
}

// Add an event listener to the 'Clear Chat' button to handle chat clearing
document.getElementById('clear-chat').addEventListener('click', function() {
    fetch('/clear_chat', {
        method: 'POST' // Send a POST request to clear the chat
    })
    .then(response => response.json()) // Process the response as JSON
    .then(data => {
        document.getElementById('chat-content').innerHTML = ''; // Clear the chat content in the UI
        scrollToBottom(); // Scroll to the bottom after clearing the chat
    });
});

// Add an event listener to the 'Generate Email' button to generate an email from the last bot message
document.getElementById('generate-email').addEventListener('click', function() {
    const emailMessageElement = addMessageToChat('Bot', '', 'bot-message'); // Add a placeholder for the email

    // Send a request to generate the email
    fetch('/generate_email', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        }
    })
    .then(response => {
        const reader = response.body.getReader(); // Get a reader for the streamed email response
        const decoder = new TextDecoder('utf-8'); // Decode the response as UTF-8
        let emailMessage = ''; // Variable to store the email message as it streams

        // Function to process the streamed email response
        function read() {
            reader.read().then(({ done, value }) => {
                if (done) {
                    updateBotMessage(emailMessageElement, emailMessage); // Update the chat with the email
                    addCopyButton(emailMessageElement); // Add a copy button to the email
                    scrollToBottom(); // Scroll to the bottom after the final email is generated
                    return;
                }
                const chunk = decoder.decode(value); // Decode the chunk
                const lines = chunk.split('\n\n'); // Split the chunk by line breaks
                lines.forEach(line => {
                    if (line.startsWith('data: ')) {
                        const token = line.replace('data: ', ''); // Remove the 'data: ' prefix
                        emailMessage += token; // Append the token to the email message
                        updateBotMessage(emailMessageElement, emailMessage); // Update the email message
                        scrollToBottom(); // Scroll during email generation
                    }
                });
                read(); // Read the next chunk
            });
        }
        read();
    })
    .catch(error => {
        updateBotMessage(emailMessageElement, 'Error generating email.'); // Show error if request fails
        scrollToBottom(); // Ensure scroll after error
    });
});
