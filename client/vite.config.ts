import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react-swc';

const cherryPickedKeys = [
  'REACT_APP_SPEECH_API_KEY',
  'REACT_APP_SPEECH_API_ENDPOINT',
  'REACT_APP_SPEECH_API_REGION',
  'REACT_APP_API_BASE_URL',
];

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  const processEnv = {};
  cherryPickedKeys.forEach((key) => (processEnv[key] = env[key]));

  return {
    define: {
      'process.env': processEnv,
    },
    plugins: [react()],
  };
});
