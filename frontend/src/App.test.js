import { render, screen, waitFor } from '@testing-library/react';
import App from './App';

beforeEach(() => {
  delete window.SpeechRecognition;
  delete window.webkitSpeechRecognition;
  global.fetch = jest.fn((url) => Promise.resolve({
    ok: true,
    json: () => Promise.resolve(String(url).includes('/translate')
      ? { translated_text: 'translated final sentence', retrieved_context: [], sources_used: [], history: [], provider: 'test', metrics: {} }
      : { llm_status: { active_mode: 'Local Multilingual Engine' } }),
  }));
});

test('renders one deterministic microphone path', async () => {
  render(<App />);
  expect(screen.getByText(/Speech & Transcript/i)).toBeInTheDocument();
  expect(screen.getByText(/record with AI speech recognition/i)).toBeInTheDocument();
  await waitFor(() => expect(global.fetch).toHaveBeenCalled());
});
