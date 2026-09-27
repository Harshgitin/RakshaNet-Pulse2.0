const fs = require('fs');
const path = require('path');

const root = path.resolve(__dirname, '..');
const required = [
  'server.js',
  'index.html',
  'js/script.js',
  'js/location.js',
  'pages/intelligence.html',
  'pages/map.html',
  'pages/emergency.html',
  'ml/landslide_model.onnx',
  'ml/landslide_model.pkl',
  'ml/model_profile.json',
  'ml/model_validation_report.json',
  'sw.js'
];

let ok = true;
for (const file of required) {
  const exists = fs.existsSync(path.join(root, file));
  console.log(`${exists ? 'OK  ' : 'MISS'} ${file}`);
  if (!exists) ok = false;
}

try {
  const profile = JSON.parse(
    fs.readFileSync(path.join(root, 'ml/model_profile.json'), 'utf8')
  );
  const featureCount = Object.keys(profile.featureImportance || {}).length;
  const quantileCount = Object.keys(profile.featureQuantiles || {}).length;
  console.log(`OK   model profile: ${featureCount} importances / ${quantileCount} quantile profiles`);
  if (featureCount !== quantileCount || featureCount !== 16) ok = false;
} catch (error) {
  console.error(`FAIL model profile: ${error.message}`);
  ok = false;
}

console.log(ok ? '\nRakshaNet project validation passed.' : '\nRakshaNet project validation found an issue.');
process.exitCode = ok ? 0 : 1;
