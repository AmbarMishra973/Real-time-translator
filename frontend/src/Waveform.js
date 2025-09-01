import React, { useEffect, useRef } from 'react';
import WaveSurfer from 'wavesurfer.js';

const Waveform = ({ audioBlob, playing, onFinish }) => {
  const waveformRef = useRef(null);
  const wavesurfer = useRef(null);

  useEffect(() => {
    if (!waveformRef.current) return;

    // Initialize WaveSurfer
    wavesurfer.current = WaveSurfer.create({
      container: waveformRef.current,
      waveColor: '#ddd',
      progressColor: '#3b82f6',
      cursorColor: '#3b82f6',
      height: 80,
      responsive: true,
      normalize: true,
    });

    // When playback finishes
    wavesurfer.current.on('finish', () => {
      if (onFinish) onFinish();
    });

    return () => {
      if (wavesurfer.current) {
        wavesurfer.current.destroy();
      }
    };
  }, [onFinish]);

  useEffect(() => {
    if (!wavesurfer.current || !audioBlob) return;

    // Load audio blob into wavesurfer
    const fileUrl = URL.createObjectURL(audioBlob);
    wavesurfer.current.load(fileUrl);

    // Cleanup URL object when component unmounts or audioBlob changes
    return () => {
      URL.revokeObjectURL(fileUrl);
    };
  }, [audioBlob]);

  useEffect(() => {
    if (!wavesurfer.current) return;
    if (playing) {
      wavesurfer.current.play();
    } else {
      wavesurfer.current.pause();
    }
  }, [playing]);

  return <div ref={waveformRef} style={{ width: '100%' }} />;
};

export default Waveform;
