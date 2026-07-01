
    window.dash_clientside = Object.assign({}, window.dash_clientside, {
        clientside: {
            button_feedback: function(...args) {
                const ctx = dash_clientside.callback_context;
                if (ctx.triggered && ctx.triggered.length > 0) {
                    const triggeredButton = document.querySelector('[data-dash-is-loading]');
                    if (triggeredButton) {
                        triggeredButton.classList.add('clicked');
                        setTimeout(() => {
                            triggeredButton.classList.remove('clicked');
                        }, 250);
                    }
                }
                return '';
            }
        }
    });
    