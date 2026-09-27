// =====================================================
// RAKSHANET ADMIN LOGIN
// =====================================================

const adminForm =
    document.getElementById(
        "adminLoginForm"
    );

const adminIdentifier =
    document.getElementById(
        "adminIdentifier"
    );

const adminPassword =
    document.getElementById(
        "adminPassword"
    );

const adminButton =
    document.getElementById(
        "adminLoginButton"
    );

const adminMessage =
    document.getElementById(
        "adminLoginMessage"
    );


function showAdminMessage(
    message,
    type = "danger"
) {

    if (!adminMessage) {
        return;
    }

    adminMessage.className =
        `rn-alert-banner ${type}`;

    adminMessage.textContent =
        message;

    adminMessage.style.display =
        "block";
}


adminForm?.addEventListener(
    "submit",
    async event => {

        event.preventDefault();


        const identifier =
            adminIdentifier
                .value
                .trim();

        const password =
            adminPassword
                .value;


        if (!identifier || !password) {

            showAdminMessage(
                "Enter your mobile/email and password."
            );

            return;
        }


        adminButton.disabled =
            true;

        adminButton.textContent =
            "⏳ VERIFYING ADMIN...";


        if (adminMessage) {
            adminMessage.style.display =
                "none";
        }


        try {

            const response =
                await fetch(
                    "/api/admin/login",
                    {
                        method: "POST",

                        headers: {
                            "Content-Type":
                                "application/json"
                        },

                        body:
                            JSON.stringify({
                                identifier,
                                password
                            })
                    }
                );


            let data = null;

            try {
                data =
                    await response.json();
            } catch (_) {
                data = null;
            }


            if (!response.ok) {

                throw new Error(
                    data?.error ||
                    `Login failed (${response.status})`
                );

            }


            if (
                !data?.ok ||
                !data?.token ||
                data?.user?.role !== "ADMIN"
            ) {

                throw new Error(
                    "Administrator authentication failed."
                );

            }


            // -------------------------------------------------
            // Store admin session
            // -------------------------------------------------

            sessionStorage.setItem(
                "rakshanet_admin_token",
                data.token
            );


            sessionStorage.setItem(
                "rakshanet_admin_user",
                JSON.stringify(
                    data.user
                )
            );


            // -------------------------------------------------
            // Go to dashboard
            // -------------------------------------------------

            window.location.href =
                "dashboard.html";

        } catch (error) {

            console.error(
                "Admin login failed:",
                error
            );


            showAdminMessage(
                error.message
            );

        } finally {

            adminButton.disabled =
                false;

            adminButton.textContent =
                "🔐 LOGIN AS ADMIN";

        }

    }
);