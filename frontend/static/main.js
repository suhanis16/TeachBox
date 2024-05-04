/**
 * Returns the current datetime for the message creation.
 */
function getCurrentTimestamp() {
	return new Date();
}

/**
 * Renders a message on the chat screen based on the given arguments.
 * This is called from the `showUserMessage` and `showBotMessage`.
 */
function renderMessageToScreen(args) {
    // local variables
    let displayDate = (args.time || getCurrentTimestamp()).toLocaleString('en-IN', {
        month: 'short',
        day: 'numeric',
        hour: 'numeric',
        minute: 'numeric',
    });
    let messagesContainer = $('.messages');

    // init element
    let message = $(`
    <li class="message ${args.message_side}">
        <div class="avatar"></div>
        <div class="text_wrapper">
            <div class="text_container">${args.text}</div>
            <div class="timestamp">${displayDate}</div>
        </div>
    </li>
    `);

    // add to parent
    messagesContainer.append(message);

    // animations
    setTimeout(function () {
        message.addClass('appeared');
    }, 0);
    messagesContainer.animate({ scrollTop: messagesContainer.prop('scrollHeight') }, 300);
}

/* Sends a message when the 'Enter' key is pressed.
 */
$(document).ready(function() {
    $('#msg_input').keydown(function(e) {
        // Check for 'Enter' key
        if (e.key === 'Enter') {
            // Prevent default behaviour of enter key
            e.preventDefault();
			// Trigger send button click event
            $('#send_button').click();
        }
    });
});

/**
 * Displays the user message on the chat screen. This is the right side message.
 */
function showUserMessage(message, datetime) {
	renderMessageToScreen({
		text: message,
		time: datetime,
		message_side: 'right',
	});
}

/**
 * Displays the chatbot message on the chat screen. This is the left side message.
 */
// function showBotMessage(message, datetime) {
// 	renderMessageToScreen({
// 		text: message,
// 		time: datetime,
// 		message_side: 'left',
// 	});
// }

function showBotMessage(message, datetime) {
    // Parse the message to format it
    const formattedMessage = formatBotResponse(message);
	console.log(formattedMessage);
    // Display the formatted message
    renderMessageToScreen({
        text: formattedMessage,
        time: datetime,
        message_side: 'left',
    });
}

function formatBotResponse(message) {
    // Split the message into paragraphs
    const paragraphs = message.split('\n\n');
    // Format each paragraph
    const formattedParagraphs = paragraphs.map(paragraph => {
        // Replace **text** with bold HTML tags
        paragraph = paragraph.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        return `<p>${paragraph}</p>`; // Wrap each paragraph in <p> tags
    });

    // Join formatted paragraphs with newline
    let formattedMessage = formattedParagraphs.join('\n');

    // Replace list markers with proper HTML list tags
    formattedMessage = formattedMessage.replace(/<p>(?:\d+.)? (.*?)<\/p>/gm, '<li>$1</li>');
    formattedMessage = formattedMessage.replace(/<\/p>\n<li>/gm, '<li>'); // Fix formatting for first list item
    formattedMessage = formattedMessage.replace(/<\/p>/gm, ''); // Remove remaining </p> tags

    // Wrap the list in <ul> tags
    formattedMessage = `<ul>${formattedMessage}</ul>`;

    return formattedMessage;
}


/**
 * Get input from user and show it on screen on button click.
 */
$('#send_button').on('click', function (e) {
	var userInput = document.getElementById('msg_input').value;
	// get and show message and reset input
	showUserMessage($('#msg_input').val());
	$('#msg_input').val('');
	var request = new XMLHttpRequest();
	request.open('POST', 'http://localhost:5000/chat', true);
	request.setRequestHeader('Content-Type', 'application/json');
	request.onload = function() {
		if (request.status >= 200 && request.status < 400) {
			var data = JSON.parse(request.responseText);
			showBotMessage(data.bot_response);
		} else {
			console.error('Error');
		}
	};
	request.onerror = function() {
		console.error('Connection error');
	};
	console.log(JSON.stringify({ user_input: userInput }))
	request.send(JSON.stringify({ user_input: userInput }));

	// show bot message
	// setTimeout(function () {
	// 	showBotMessage(randomstring());
	// }, 300);
});

/**
 * Returns a random string. Just to specify bot message to the user.
 */
function randomstring(length = 20) {
	let output = '';

	// magic function
	var randomchar = function () {
		var n = Math.floor(Math.random() * 62);
		if (n < 10) return n;
		if (n < 36) return String.fromCharCode(n + 55);
		return String.fromCharCode(n + 61);
	};

	while (output.length < length) output += randomchar();
	return output;
}

/**
 * Set initial bot message to the screen for the user.
 */

const formattedMessage = `
        <p><strong>Hello! I'm your STEM teaching assistant chatbot.</strong> I'm excited to help you create dynamic and effective lesson plans! Let's personalize your teaching approach.</p>
        <p><strong>To get started, please provide the following information:</strong></p>
        <ul>
            <li><strong>Engineering Major:</strong> (e.g., Mechanical Engineering, Computer Science, etc.)</li>
            <li><strong>Student Level:</strong> (e.g., Freshman, Junior, Graduate)</li>
            <li><strong>Class Size</strong></li>
			<li><strong>Class Duration</strong></li>
            <li><strong>Teaching Mode:</strong> (Online or Offline)</li>
            <li><strong>Topic</strong></li>
        </ul>
        <p>With this information, I can suggest tailored active learning strategies, provide credible resources, and even offer relevant citations. Let's transform your classroom experience together!</p>
        <p><strong>Here's why I may be helpful:</strong></p>
        <ul>
            <li><strong>Active Learning:</strong> I'll help you break away from passive lectures.</li>
            <li><strong>Customization:</strong> I understand your unique teaching context.</li>
            <li><strong>Resourceful:</strong> I'll source reliable materials to support your lessons.</li>
        </ul>
        <p><strong>Are you ready for a more engaging and effective teaching experience? Let's get started!</strong></p>
    `;

$(window).on('load', function () {
showBotMessage(formattedMessage);
});
