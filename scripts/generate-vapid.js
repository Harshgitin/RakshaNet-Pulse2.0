const webPush = require('web-push');

const keys = webPush.generateVAPIDKeys();
console.log('\nRakshaNet Web Push VAPID keys\n');
console.log(`VAPID_SUBJECT=mailto:alerts@example.com`);
console.log(`VAPID_PUBLIC_KEY=${keys.publicKey}`);
console.log(`VAPID_PRIVATE_KEY=${keys.privateKey}`);
console.log('\nCopy these values into .env. Keep VAPID_PRIVATE_KEY secret.\n');
