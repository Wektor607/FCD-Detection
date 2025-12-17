import React, { useRef } from 'react';

function FileInput({ onFileChange, file, showSelected }) {
  const inputRef = useRef(null);

  const handleChange = (e) => {
    onFileChange(e);
    // input value is not reset, so user cannot select the same file twice in a row
  };

  return (
    <div className="form-actions">
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', minWidth: 180 }}>
        <label className="upload-label">
          <span className="btn small-btn cursor-pointer">📂 Choose File</span>
          <input
            type="file"
            accept=".hdf5"
            onChange={handleChange}
            required
            className="hidden"
            ref={inputRef}
          />
        </label>
        {showSelected && file && (
          <span className="file-selected fade-in_file" style={{ marginTop: 4 }}>
            ✅ Selected: {file.name}
          </span>
        )}
      </div>
    </div>
  );
}

export default FileInput;
