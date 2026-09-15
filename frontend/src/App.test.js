import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
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

test('renders the translator with Whisper as the visible deterministic STT option', async () => {
  render(<App />);
  expect(screen.getByText(/Speech & Transcript/i)).toBeInTheDocument();
  expect(screen.getByText(/record with Whisper AI/i)).toBeInTheDocument();
  await waitFor(() => expect(global.fetch).toHaveBeenCalled());
});

test('Live Mic translates only after a final recognition result ends the session', async () => {
  class MockRecognition {
    static latest;
    constructor() { MockRecognition.latest = this; }
    start = jest.fn();
    stop = jest.fn();
  }
  window.SpeechRecognition = MockRecognition;

  render(<App />);
  await waitFor(() => expect(global.fetch).toHaveBeenCalledTimes(1));

  fireEvent.click(screen.getByRole('button', { name: /Live Mic/i }));
  fireEvent.click(screen.getByTitle('Click to Speak'));
  fireEvent.click(screen.getByTitle('Stop & Process'));
  expect(MockRecognition.latest.stop).toHaveBeenCalledTimes(1);
  expect(global.fetch).toHaveBeenCalledTimes(1);

  await act(async () => {
    MockRecognition.latest.onresult({
      resultIndex: 0,
      results: [{ isFinal: true, 0: { transcript: 'final sentence' } }],
    });
    MockRecognition.latest.onend();
    await Promise.resolve();
  });

  await waitFor(() => expect(global.fetch).toHaveBeenCalledTimes(2));
  expect(screen.getByDisplayValue('final sentence')).toBeInTheDocument();
});
