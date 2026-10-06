import { connection } from "next/server";
import { getSketch } from "@/lib/db.ts";
import { GENERAL_SKETCH } from "@/lib/sketch.ts";
import { GeneralSketchPad } from "@/components/GeneralSketchPad";

export default async function SketchpadPage() {
  await connection();
  return (
    <main className="flex flex-col gap-3 p-4 lg:h-full lg:px-6">
      <header className="grid gap-1">
        <p className="eyebrow">Sketchpad</p>
        <p className="text-sm text-muted">A scratch space for anything. On an exercise page, the Sketchpad link opens a sketch saved with that exercise instead.</p>
      </header>
      <div className="min-h-0 flex-1">
        <GeneralSketchPad id={GENERAL_SKETCH} initialScene={getSketch(GENERAL_SKETCH)} />
      </div>
    </main>
  );
}
