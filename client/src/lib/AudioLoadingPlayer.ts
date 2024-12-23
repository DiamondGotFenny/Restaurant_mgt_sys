export class AudioLoadingPlayer {
  private audioContext: AudioContext | null = null;
  private sourceNode: AudioBufferSourceNode | null = null;
  private gainNode: GainNode | null = null;
  private audioBuffer: AudioBuffer | null = null;
  private isPlaying: boolean = false;
  private intervalId: NodeJS.Timeout | null = null;

  constructor() {
    this.loadAudioBuffer();
  }

  private async loadAudioBuffer() {
    try {
      const response = await fetch(
        `${process.env.REACT_APP_API_BASE_URL}/static-loading-audio/`
      );
      const arrayBuffer = await response.arrayBuffer();
      this.audioContext = new AudioContext();
      this.audioBuffer = await this.audioContext.decodeAudioData(arrayBuffer);
    } catch (error) {
      console.error('Error loading audio buffer:', error);
    }
  }

  public start() {
    if (!this.audioBuffer || !this.audioContext || this.isPlaying) return;

    this.isPlaying = true;
    this.playLoadingSound();

    this.intervalId = setInterval(() => {
      this.playLoadingSound();
    }, this.audioBuffer.duration * 1000 + 500); // Add 500ms gap between plays
  }

  public stop() {
    this.isPlaying = false;
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = null;
    }
    if (this.sourceNode) {
      this.sourceNode.stop();
      this.sourceNode.disconnect();
      this.sourceNode = null;
    }
  }

  private playLoadingSound() {
    if (!this.isPlaying || !this.audioBuffer || !this.audioContext) return;

    this.sourceNode = this.audioContext.createBufferSource();
    this.sourceNode.buffer = this.audioBuffer;

    this.gainNode = this.audioContext.createGain();
    this.sourceNode.connect(this.gainNode);
    this.gainNode.connect(this.audioContext.destination);

    this.sourceNode.onended = () => {
      if (this.sourceNode) {
        this.sourceNode.disconnect();
        this.sourceNode = null;
      }
    };

    this.sourceNode.start();
  }

  public setVolume(value: number) {
    if (this.gainNode) {
      this.gainNode.gain.setValueAtTime(
        value,
        this.audioContext?.currentTime || 0
      );
    }
  }
}
