import React, { useState } from "react";
import BackButton from "../components/UploadForm/BackButton";
import DropdownModel from "../components/UploadForm/DropdownModel";
import FileInput from "../components/UploadForm/FileInput";
import ResultPreview from "../components/UploadForm/ResultPreview";
import LoadingCat from "../components/UploadForm/Loading";

function UploadPage() {
  const [file, setFile] = useState(null);
  const [description, setDescription] = useState("");
  const [modelType, setModelType] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showSelected, setShowSelected] = useState(false);

  const models = [
    { name: "MELD", desc: "Original MELD model"},
    { name: "Exp1", desc: "MELD frozen; trained Decoder only (HexUnpool + SpiralConv + head)" },
    { name: "Exp2", desc: "MELD frozen; trained Decoder + self-attention on vision features" },
    { name: "Exp3_full_text", desc: "MELD frozen; trained Decoder + self-attention + cross-attention to full text encoder" },
    { name: "Exp3_lobe+hemi", desc: "As Exp3_full_text, but text limited to lobe + hemisphere names" },
  ];

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("description", description);
    formData.append("model_type", modelType);

    try {
      const res = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      setResult(data);
    } catch (err) {
      console.error("Error:", err);
      alert("Upload failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);

    if (selectedFile) {
      setShowSelected(true);
      setTimeout(() => setShowSelected(false), 10000);
    }
  };

  return (
    <div className="upload-page">
      <div className="card">
        <BackButton />

        <h1 className="text-2xl font-bold text-center mb-6">
          Upload HDF5 File
        </h1>

        <form onSubmit={handleSubmit} className="space-y-4">
          {/* <div style={{color: 'red', fontSize: '0.95em', marginBottom: '0.5em'}}>
            <div>DEBUG: file = {file ? file.name : 'null'}</div>
            <div>DEBUG: modelType = {modelType || 'null'}</div>
          </div> */}
          <div className="form-field">
            <label htmlFor="description" className="field-label">
              Description
            </label>
            <textarea
              id="description"
              className="upload-textarea"
              placeholder="Enter description..."
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              rows="3"
            />
          </div>

          <DropdownModel
            models={models}
            modelType={modelType}
            setModelType={setModelType}
          />

          <FileInput
            onFileChange={handleFileChange}
            file={file}
            showSelected={showSelected}
          />

          <button
            className="btn small-btn"
            type="submit"
            disabled={loading || !modelType || !file}
          >
            {loading ? <LoadingCat /> : "Send"}
          </button>
        </form>

        {result && <ResultPreview result={result} file={file} />}
      </div>
    </div>
  );
}

export default UploadPage;
