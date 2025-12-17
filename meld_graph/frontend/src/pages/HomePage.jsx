import { Link } from "react-router-dom";
import "../styles.css";
import LogoImage from "../assets/logo.png"; // логотип с прозрачным фоном

function HomePage() {
  return (
    <div>
      {/* Хедер */}
      <header className="header">
        <img src={LogoImage} alt="Epilepsy Detector Logo" className="header-logo" />
        {/* <span className="header-title">Epilepsy Detector</span> */}
      </header>

      {/* Контент */}
      <main className="home-container">
        <div className="card">
          <h1>About the project</h1>

          <p>
            Numerous methods have been developed for the detection of tumors in
            internal organs such as the lungs, brain, kidneys, and breast. However,
            detecting epileptogenic lesions remains significantly more challenging.
            Unlike tumors, these lesions do not typically increase in size over
            time, and there is a severe shortage of publicly available annotated
            datasets. As a result, researchers often need to contact hospitals and
            clinical centers directly to obtain even a minimal number of scans.
            Although recent studies have demonstrated that deep learning models can
            detect epileptogenic lesions, their performance remains limited,
            highlighting the ongoing difficulty of this task.
          </p>

          <p>
            At the same time, recent work on tumor detection using text-guided
            approaches has shown promising results, demonstrating that incorporating
            textual descriptions can significantly improve segmentation accuracy.
            Inspired by these advances, this study proposes a new method that
            combines visual and textual features to enhance FCD detection under
            limited data conditions. We further present a systematic comparison of
            multiple types of textual annotations and analyze their influence on
            model performance.
          </p>

          <div className="button-row">
            <a
              href="https://github.com/Wektor607/FCD-Detection/blob/master/meld_graph/README.md"
              className="btn"
            >
              Documentation
            </a>

            <Link to="/upload" className="btn">
              Get Prediction
            </Link>
          </div>
        </div>
      </main>
    </div>
  );
}

export default HomePage;
