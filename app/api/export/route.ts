import { exportAll } from "@/lib/db.ts";
import { localToday } from "@/lib/progress.ts";

export async function GET() {
  return new Response(JSON.stringify({ exportedAt: new Date().toISOString(), ...exportAll() }, null, 2), {
    headers: {
      "Content-Type": "application/json",
      "Content-Disposition": `attachment; filename="mentora-backup-${localToday()}.json"`,
      "Cache-Control": "no-store",
    },
  });
}
