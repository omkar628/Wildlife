export type DesktopBridge = {
  isElectron: boolean;
  selectFolder: () => Promise<string | null>;
};

declare global {
  interface Window {
    desktop?: DesktopBridge;
  }
}

export function isDesktopApp(): boolean {
  return Boolean(window.desktop?.isElectron);
}

export async function selectCameraFolder(): Promise<string | null> {
  if (!window.desktop?.selectFolder) {
    throw new Error(
      "Native folder picker is only available in the desktop app. From the project folder run: npm run dev",
    );
  }
  return window.desktop.selectFolder();
}

export function suggestCameraId(folderPath: string): string {
  const trimmed = folderPath.replace(/[\\/]+$/, "");
  const parts = trimmed.split(/[\\/]/).filter(Boolean);
  const last = parts[parts.length - 1] ?? "";
  return /^[A-Za-z0-9._-]+$/.test(last) ? last : "";
}
