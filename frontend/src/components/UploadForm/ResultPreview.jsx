import { ArrowDown } from "lucide-react";

function ResultPreview({ result, file }) {
  return (
    <div className="fade-in mt-8 text-center">
      <h2 className="text-2xl font-bold text-left mb-4">Result</h2>
      <div className="result-text" style={{margin: '0 auto 18px auto', display: 'inline-block'}}>
        {result.text}
      </div>
      <div style={{ position: 'relative', display: 'inline-block', marginBottom: '8px' }}>
        <img
          src={`http://localhost:8000${result.result_png}?t=${Date.now()}`}
          alt="Model result"
          className="result-image"
        />
      </div>

      <div className="download-buttons">
        <a
          href={`http://localhost:8000${result.download_png}`}
          className="download-btn blue"
        >
          <ArrowDown size={18} />
          Download PNG
        </a>

        <a
          href={`http://localhost:8000${result.download_nii}`}
          className="download-btn green"
        >
          <ArrowDown size={18} />
          Download NIfTI
        </a>
      </div>
    </div>
  );
}

export default ResultPreview;
