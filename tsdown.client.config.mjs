import { fileURLToPath } from 'node:url'
import manifest from './package.json' with { type: 'json' }

export default {
  entry: { client: fileURLToPath(new URL('./src/client/index.tsx', import.meta.url)) },
  outDir: fileURLToPath(new URL('./dist/client', import.meta.url)),
  tsconfig: fileURLToPath(new URL('./tsconfig.client.json', import.meta.url)),
  format: 'cjs',
  platform: 'browser',
  target: 'es2022',
  clean: false,
  dts: false,
  deps: {
    neverBundle: ['react', 'react/jsx-runtime', 'react-dom', 'react-dom/client', '@deepseek-ai/cordis', '@dsh-plugin/dsh-loader/client'],
  },
  outputOptions: {
    entryFileNames: 'client.js',
    banner: `window.__ModuleLoader__.load({ id: ${JSON.stringify(manifest.name)}, factory: (require) => {`,
    intro: 'var module = { exports: {} }; var exports = module.exports;',
    footer: 'return module.exports; } });',
    codeSplitting: false,
  },
}
