import { ArrowDown } from "lucide-react";
import NiiVueViewer from "./NiiVueViewer";

function ResultPreview({ result, file }) {
  return (
    <div className="fade-in mt-8 text-center">
      <h2 className="text-2xl font-bold text-left mb-4">Result</h2>
      
      {/* TEXT */}
      <div className="result-text" style={{margin: '0 auto 18px auto', display: 'inline-block'}}>
        {result.text}
      </div>
      
      {/* IMAGES: column */}
      {/* <div style={{ position: 'relative', display: 'inline-block', marginBottom: '8px' }}> */}
      <div className="result-images-row">
        <div className="result-image-block">
          <div className="result-image-title">2D MRI slices</div>
          <img
            src={`http://localhost:8000${result.result_2dpng}?t=${Date.now()}`}
            alt="2D MRI"
            className="result-image"
          />
        </div>

        <div className="result-image-block">
          <div className="result-image-title">3D cortical surface</div>
          <img
            src={`http://localhost:8000${result.result_3dpng}?t=${Date.now()}`}
            alt="3D surface"
            className="result-image"
          />
        </div>
      </div>

      {/* INTERACTIVE NIFTI */}
      {result.result_nii && result.t1_bg && (
        <div className="mt-10">
          <h3 className="text-xl font-semibold text-left mb-3">
            Interactive 3D NIfTI
          </h3>

          <NiiVueViewer
            t1Url={result.t1_bg}
            predUrl={result.result_nii}
          />
        </div>
      )}

      {/* BUTTONS: row */}
      <div className="download-buttons">
        <a
          href={`http://localhost:8000${result.download_2dpng}`}
          className="download-btn blue"
        >
          <ArrowDown size={18} />
          Download 2D PNG
        </a>

        <a
          href={`http://localhost:8000${result.download_3dpng}`}
          className="download-btn blue"
        >
          <ArrowDown size={18} />
          Download 3D PNG
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
