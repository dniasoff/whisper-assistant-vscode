import * as vscode from 'vscode';
import { exec, ChildProcess, spawn } from 'child_process';
import * as fs from 'fs';
import * as path from 'path';
import { promisify } from 'util';
import OpenAI from 'openai';
import { ctrlc } from 'ctrlc-windows';

const execAsync = promisify(exec);

interface Segment {
  id: number;
  seek: number;
  start: number;
  end: number;
  text: string;
  tokens: number[];
  temperature: number;
}

export interface Transcription {
  text: string;
  segments: Segment[];
  language: string;
}

export type WhisperModel = 'whisper-1' | 'whisper-large-v3-turbo';

type ApiProvider = 'localhost' | 'openai' | 'groq';

interface ApiConfig {
  baseURL: string;
  apiKey: string;
}

const PROVIDER_MODELS: Record<ApiProvider, WhisperModel> = {
  openai: 'whisper-1',
  groq: 'whisper-large-v3-turbo',
  localhost: 'whisper-1', // default to OpenAI model for localhost
};

class SpeechTranscription {
  private fileName: string = 'recording';
  private recordingProcess: ChildProcess | null = null;
  private tempDir: string;
  private isUsingSox: boolean = false;

  constructor(
    private storagePath: string,
    private outputChannel: vscode.OutputChannel,
  ) {
    // Create a temp directory within the storage path
    this.tempDir = path.join(this.storagePath, 'temp');
    if (!fs.existsSync(this.tempDir)) {
      fs.mkdirSync(this.tempDir, { recursive: true });
    }
  }

  private getApiConfig(): ApiConfig {
    const config = vscode.workspace.getConfiguration('whisper-assistant');
    const provider = config.get<ApiProvider>('apiProvider') || 'localhost';

    const apiKey = config.get<string>('apiKey');
    if (!apiKey) {
      vscode.window.showErrorMessage(
        `Whisper Assistant: API key not configured for ${provider}. Please set the API key in settings.`,
      );
      throw new Error(`API key not configured for ${provider}`);
    }

    const baseURLs: Record<ApiProvider, string> = {
      localhost:
        (config.get('customEndpoint') || 'http://localhost:4444') + '/v1',
      openai: 'https://api.openai.com/v1',
      groq: 'https://api.groq.com/openai/v1',
    };

    return {
      baseURL: baseURLs[provider],
      apiKey,
    };
  }

  async checkIfInstalled(command: string): Promise<boolean> {
    try {
      await execAsync(`${command} --help`);
      return true;
    } catch (error) {
      return false;
    }
  }

  getOutputDir(): string {
    return this.storagePath;
  }

  startRecording(): boolean {
    try {
      // Ensure temp directory exists
      if (!fs.existsSync(this.tempDir)) {
        fs.mkdirSync(this.tempDir, { recursive: true });
      }

      const outputPath = path.join(this.tempDir, `${this.fileName}.wav`);

      // Check for custom recording command first
      const config = vscode.workspace.getConfiguration('whisper-assistant');
      const customCommand = config.get<string>('customRecordingCommand') || '';

      if (customCommand.trim()) {
        this.outputChannel.appendLine(
          'Whisper Assistant: Using custom recording command',
        );
        return this.startCustomRecording(outputPath, customCommand);
      } else {
        this.outputChannel.appendLine(
          'Whisper Assistant: Using sox for recording (default)',
        );
        this.startSoxRecording(outputPath);
        return true;
      }
    } catch (error) {
      this.outputChannel.appendLine(`Whisper Assistant: error: ${error}`);
      return false;
    }
  }

  private startCustomRecording(
    outputPath: string,
    customCommand: string,
  ): boolean {
    // Validate that the command contains the required placeholder
    if (!customCommand.includes('$AUDIO_FILE')) {
      vscode.window.showErrorMessage(
        'Whisper Assistant: Custom recording command must include $AUDIO_FILE placeholder. Example: ffmpeg -f avfoundation -i :1 -ac 1 -ar 16000 -sample_fmt s16 $AUDIO_FILE',
      );
      return false; // Return false to indicate failure
    }

    // Detect if custom command is using sox
    const commandLower = customCommand.toLowerCase().trim();
    this.isUsingSox = commandLower.startsWith('sox ') || commandLower === 'sox';

    // Replace the placeholder with the actual output path (properly quoted)
    const command = customCommand.replace(/\$AUDIO_FILE/g, `"${outputPath}"`);

    this.outputChannel.appendLine(
      `Whisper Assistant: Executing custom command: ${command}`,
    );

    // Use platform-appropriate shell execution
    this.recordingProcess = this.spawnWithShell(command);
    this.setupProcessHandlers();
    return true; // Return true to indicate success
  }

  private spawnWithShell(command: string): ChildProcess {
    const platform = process.platform;

    switch (platform) {
      case 'win32':
        // Use PowerShell on Windows to properly handle special characters like parentheses
        return spawn('powershell.exe', ['-NoProfile', '-Command', command], {
          stdio: ['pipe', 'pipe', 'pipe'],
        });

      case 'darwin':
      case 'linux':
      default:
        // Use sh on Unix-like systems (macOS, Linux, etc.)
        return spawn('sh', ['-c', command], {
          stdio: ['pipe', 'pipe', 'pipe'],
        });
    }
  }

  private startSoxRecording(outputPath: string): void {
    try {
      this.isUsingSox = true;
      this.recordingProcess = spawn(
        'sox',
        [
          '-d',
          '-b',
          '16',
          '-e',
          'signed',
          '-c',
          '1',
          '-r',
          '16k',
          outputPath,
        ],
        {
          stdio: ['pipe', 'pipe', 'pipe'],
        },
      );
      this.setupProcessHandlers();
    } catch (error) {
      this.outputChannel.appendLine(
        `Whisper Assistant: Error starting sox recording: ${error}`,
      );
      vscode.window.showErrorMessage(
        'Whisper Assistant: Failed to start sox recording. Please ensure sox is installed.',
      );
      throw error; // Re-throw so startRecording() can return false
    }
  }

  private setupProcessHandlers(): void {
    if (this.recordingProcess) {
      // Only show initial configuration
      let initialConfigShown = false;
      let stderrBuffer = '';

      this.recordingProcess.stderr?.on('data', (data) => {
        const message = data.toString();
        stderrBuffer += message;
        // Only show the initial configuration message
        if (!initialConfigShown && message.includes('Input File')) {
          this.outputChannel.appendLine(
            `Whisper Assistant: Configuration: ${message.trim()}`,
          );
          initialConfigShown = true;
        }
      });

      this.recordingProcess.stdout?.on('data', (data) => {
        this.outputChannel.appendLine(`Whisper Assistant: stdout: ${data}`);
      });

      this.recordingProcess.on('close', (code) => {
        // On Windows, CTRL-C exit code is 4294967295 (unsigned representation of -1)
        const expectedExitCode = process.platform === 'win32' ? 4294967295 : null;
        if (code === 0 || code === expectedExitCode) {
          this.outputChannel.appendLine(
            'Whisper Assistant: Process exited successfully',
          );
        } else if (code !== null) {
          this.outputChannel.appendLine(
            `Whisper Assistant: Recording process exited with code ${code}`,
          );
          // Log stderr buffer for debugging errors
          if (stderrBuffer) {
            this.outputChannel.appendLine(
              `Whisper Assistant: Process stderr:\n${stderrBuffer}`,
            );
          }
        }
      });
    }
  }

  async stopRecording(): Promise<void> {
    if (!this.recordingProcess) {
      this.outputChannel.appendLine(
        'Whisper Assistant: No recording process found',
      );
      return;
    }

    this.outputChannel.appendLine(
      'Whisper Assistant: Stopping recording gracefully',
    );

    // Set up exit listener before terminating process
    const gracefulShutdown = new Promise<void>((resolve) => {
      const timeout = setTimeout(() => {
        resolve();
      }, 2000); // 2 second timeout

      this.recordingProcess!.once('exit', () => {
        clearTimeout(timeout);
        resolve();
      });
    });

    // On Windows, use CTRL-C to gracefully shutdown sox
    if (process.platform === 'win32' && this.isUsingSox) {
      this.outputChannel.appendLine(
        'Whisper Assistant: Sending CTRL-C to sox process',
      );
      try {
        ctrlc(this.recordingProcess.pid!);
      } catch (error) {
        this.outputChannel.appendLine(
          `Whisper Assistant: Error sending CTRL-C: ${error}`,
        );
      }

      // Give sox a moment to process the CTRL-C signal
      await new Promise((resolve) => setTimeout(resolve, 500));
    } else {
      // For ffmpeg, try sending 'q' to stdin first
      if (this.recordingProcess.stdin) {
        try {
          this.recordingProcess.stdin.write('q\n');
          this.recordingProcess.stdin.end();
        } catch (error) {
          this.outputChannel.appendLine(
            `Whisper Assistant: Error sending 'q' to stdin: ${error}`,
          );
        }
      }

      // Send SIGINT for graceful shutdown on Unix systems
      try {
        this.recordingProcess.kill('SIGINT');
      } catch (error) {
        this.outputChannel.appendLine(
          `Whisper Assistant: Error sending SIGINT: ${error}`,
        );
      }
    }

    await gracefulShutdown;

    // If process is still running (exitCode is null), force kill it
    if (
      this.recordingProcess &&
      this.recordingProcess.exitCode === null &&
      !this.recordingProcess.killed
    ) {
      this.outputChannel.appendLine(
        'Whisper Assistant: Force stopping recording process',
      );
      this.recordingProcess.kill('SIGKILL');
    }

    this.recordingProcess = null;
  }

  async transcribeRecording(): Promise<Transcription | undefined> {
    const config = vscode.workspace.getConfiguration('whisper-assistant');
    const provider = config.get<ApiProvider>('apiProvider') || 'localhost';

    let apiConfig: ApiConfig;
    try {
      apiConfig = this.getApiConfig();
    } catch (error) {
      // getApiConfig already shows a user error message, just return
      return undefined;
    }

    try {
      this.outputChannel.appendLine(
        `Whisper Assistant: Transcribing recording using ${provider} API`,
      );

      // Wait for file to be ready and have content
      const recordingPath = path.join(this.tempDir, `${this.fileName}.wav`);
      let fileReady = false;
      let attempts = 0;
      const maxAttempts = 10;

      while (!fileReady && attempts < maxAttempts) {
        try {
          const stats = fs.statSync(recordingPath);
          if (stats.size > 0) {
            fileReady = true;
          } else {
            attempts++;
            await new Promise((resolve) => setTimeout(resolve, 100));
          }
        } catch (error) {
          attempts++;
          await new Promise((resolve) => setTimeout(resolve, 100));
        }
      }

      if (!fileReady) {
        vscode.window.showErrorMessage(
          'Whisper Assistant: Recording file is not ready for transcription',
        );
        return undefined;
      }

      const audioFile = fs.createReadStream(recordingPath);

      // Get the configured model or fall back to provider default
      const configuredModel = config.get<string | null>('model');
      const model = configuredModel ?? PROVIDER_MODELS[provider];

      this.outputChannel.appendLine(
        `Whisper Assistant: Using model ${model} for ${provider}`,
      );

      const openai = new OpenAI({
        baseURL: apiConfig.baseURL,
        apiKey: apiConfig.apiKey,
        maxRetries: 0,
        timeout: 30 * 1000,
      });

      if (!openai) {
        vscode.window.showErrorMessage(
          'Whisper Assistant: Failed to initialize OpenAI client. Please check your API configuration.',
        );
        return undefined;
      }

      const transcription = await openai.audio.transcriptions.create({
        file: audioFile,
        model: model,
        language: 'en',
        // eslint-disable-next-line @typescript-eslint/naming-convention
        response_format: 'verbose_json',
      });

      // Convert response to our Transcription interface
      const result: Transcription = {
        text: transcription.text,
        segments:
          transcription.segments?.map((seg) => ({
            id: seg.id,
            seek: 0,
            start: seg.start,
            end: seg.end,
            text: seg.text,
            tokens: [],
            temperature: 0,
          })) ?? [],
        language: transcription.language,
      };

      // Save transcription to storage path
      await fs.promises.writeFile(
        path.join(this.tempDir, `${this.fileName}.json`),
        JSON.stringify(result, null, 2),
      );

      this.outputChannel.appendLine(
        `Whisper Assistant: Transcription: ${result.text}`,
      );

      if (result?.text?.length === 0) {
        return undefined;
      }

      return result;
    } catch (error) {
      // Log the error to output channel
      this.outputChannel.appendLine(
        `Whisper Assistant: error: ${error} (apiConfig.baseURL: ${apiConfig.baseURL})`,
      );

      if (provider === 'localhost') {
        vscode.window.showErrorMessage(
          'Whisper Assistant: Ensure local Whisper server is running.',
        );
      } else if (error instanceof Error) {
        // Format the error message to be more user-friendly
        const errorMessage = error.message
          .replace(/\bError\b/i, '') // Remove redundant "Error" word
          .trim();

        vscode.window.showErrorMessage(`Whisper Assistant: ${errorMessage}`);
      }

      return undefined;
    }
  }

  async deleteFiles(): Promise<void> {
    try {
      // On Windows, give the system a moment to release file handles
      if (process.platform === 'win32') {
        await new Promise((resolve) => setTimeout(resolve, 500));
      }

      const wavFile = path.join(this.tempDir, `${this.fileName}.wav`);
      const jsonFile = path.join(this.tempDir, `${this.fileName}.json`);

      if (fs.existsSync(wavFile)) {
        await fs.promises.unlink(wavFile);
      }
      if (fs.existsSync(jsonFile)) {
        await fs.promises.unlink(jsonFile);
      }
    } catch (error) {
      this.outputChannel.appendLine(
        `Whisper Assistant: Error deleting files: ${error}`,
      );
    }
  }

  // Add cleanup method for extension deactivation
  cleanup(): void {
    try {
      if (fs.existsSync(this.tempDir)) {
        fs.rmSync(this.tempDir, { recursive: true, force: true });
      }
    } catch (error) {
      this.outputChannel.appendLine(
        `Whisper Assistant: Error cleaning up: ${error}`,
      );
    }
  }
}

export default SpeechTranscription;
