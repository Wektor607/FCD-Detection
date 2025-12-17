import { ArrowLeft } from "lucide-react";

function BackButton() {
  return (
    <button
      className="btn small-btn mb-4 flex items-center gap-2"
      onClick={() => window.history.back()}
    >
      <ArrowLeft size={18} />
      Back
    </button>
  );
}

export default BackButton;
