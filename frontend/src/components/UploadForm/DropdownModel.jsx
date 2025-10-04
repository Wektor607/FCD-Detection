import { ChevronDown, Check } from "lucide-react";
import { useState } from "react";

function DropdownModel({ models, modelType, setModelType }) {
  const [isOpen, setIsOpen] = useState(false);
  const [hoveredModel, setHoveredModel] = useState(null);

  return (
    <div className="dropdown">
      <button
        type="button"
        className={`dropdown-btn ${isOpen ? "open" : ""}`}
        onClick={() => setIsOpen(!isOpen)}
      >
        {modelType || "Choose Model"}
        <ChevronDown size={18} className="chevron-icon" />
      </button>

      <div className={`dropdown-list ${isOpen ? "open" : ""}`}>
        {models.map((m) => (
          <div
            key={m.name}
            className="dropdown-item"
            onClick={() => {
              setModelType(m.name);
              setIsOpen(false);
            }}
            onMouseEnter={() => setHoveredModel(m)}
            onMouseLeave={() => setHoveredModel(null)}
          >
            <span>{m.name}</span>
            {modelType === m.name && <Check size={18} color="green" />}
          </div>
        ))}
      </div>

      <div className="model-description-wrapper">
        <div className="model-description show">
          {(hoveredModel || modelType) ? (
            <div className="description-content">
              <div className="description-header">
                <strong>ℹ️ Description:</strong>
              </div>
              <div className="description-text">
                {hoveredModel
                  ? hoveredModel.desc
                  : models.find((m) => m.name === modelType)?.desc}
              </div>
            </div>
          ) : (
            "ℹ️ Select the model to see the description"
          )}
        </div>
      </div>
    </div>
  );
}

export default DropdownModel;
