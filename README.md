# Label Studio Grounding‑DINO‑SAM Integration

> A step‑by‑step guide to setting up and running our Grounding‑DINO‑SAM backend in Label Studio.

---

## 📖 Original README

For upstream documentation and advanced configuration options, see:

[HumanSignal/label-studio-ml-backend · README.md](https://github.com/HumanSignal/label-studio-ml-backend/blob/main/README.md)

---

## 🛠️ Prerequisites

* **Python 3.7+**
* **Docker** & **Docker Compose** (ensure your user is in the `docker` group)

---

## 🚀 Installation

1. **Install Label Studio**

   ```bash
   pip install label-studio
   ```
2. **Build & launch services**

   ```bash
   docker compose up --build -d
   ```

---

## ▶️ Running the Service

* **Start the Label Studio web server**

  ```bash
  label-studio start
  ```

* **Tail logs for the Grounding‑SAM container**

  ```bash
  docker logs -f grounding_sam
  ```

---

## ⚙️ Project Configuration

1. **Log in** to Label Studio with your credentials.

2. **Create a new project**:

   * Navigate to **Projects** → **Create New Project**

3. **Import data** and choose **Labeling setup** → **Custom template**.

   <details>
   <summary>Example XML Template</summary>

   ```xml
    <View>
      <!-- your image -->
      <Image name="image" value="$image" zoom="true"/>
    
      <!-- single prompt text area -->
      <TextArea name="prompt" toName="image"
                editable="true"
                rows="2" maxSubmissions="1"
                showSubmitButton="true"/>
    
      <!-- rectangle labels -->
      <RectangleLabels name="label1" toName="image">
        <Label value="fire"             background="#FFA39E"/>
        <Label value="smoke plumes"     background="#D4380D"/>
        <Label value="hot embers"       background="#FFC069"/>
        <Label value="smoldering zones" background="#AD8B00"/>
        <Label value="tree trunks"      background="#D3F261"/>
        <Label value="industrial tank"  background="#389E0D"/>
        <Label value="industrial pipe"  background="#5CDBD3"/>
        <Label value="fences"           background="#096DD9"/>
        <Label value="debris"           background="#ADC6FF"/>
        <Label value="navigable road"   background="#9254DE"/>
        <Label value="thermal hotspots" background="#F759AB"/>
        <Label value="flooded road"     background="#FFA39E"/>
        <Label value="submerged road surface" background="#D4380D"/>
        <Label value="flood entry-points"      background="#FFC069"/>
        <Label value="drain inlets"            background="#AD8B00"/>
        <Label value="chemical leaks"          background="#D3F261"/>
        <Label value="collapsed rubble"        background="#389E0D"/>
        <Label value="damaged buildings"       background="#5CDBD3"/>
        <Label value="cracked ground"          background="#096DD9"/>
        <Label value="human"                 background="#ADC6FF"/>
        <Label value="emergency personnel"     background="#9254DE"/>
        <Label value="firetrucks"              background="#F759AB"/>
        <Label value="ambulances"              background="#FFA39E"/>
        <Label value="hazard tape"             background="#D4380D"/>
        <Label value="cones"                   background="#FFC069"/>
      </RectangleLabels>
    
      <!-- brush labels -->
      <BrushLabels name="label2" toName="image">
        <Label value="fire"             background="#FFA39E"/>
        <Label value="smoke plumes"     background="#D4380D"/>
        <Label value="hot embers"       background="#FFC069"/>
        <Label value="smoldering zones" background="#AD8B00"/>
        <Label value="tree trunks"      background="#D3F261"/>
        <Label value="industrial tank"  background="#389E0D"/>
        <Label value="industrial pipe"  background="#5CDBD3"/>
        <Label value="fences"           background="#096DD9"/>
        <Label value="debris"           background="#ADC6FF"/>
        <Label value="navigable road"   background="#9254DE"/>
        <Label value="thermal hotspots" background="#F759AB"/>
        <Label value="flooded road"     background="#FFA39E"/>
        <Label value="submerged road surface" background="#D4380D"/>
        <Label value="flood entry-points"      background="#FFC069"/>
        <Label value="drain inlets"            background="#AD8B00"/>
        <Label value="chemical leaks"          background="#D3F261"/>
        <Label value="collapsed rubble"        background="#389E0D"/>
        <Label value="damaged buildings"       background="#5CDBD3"/>
        <Label value="cracked ground"          background="#096DD9"/>
        <Label value="human"                 background="#ADC6FF"/>
        <Label value="emergency personnel"     background="#9254DE"/>
        <Label value="firetrucks"              background="#F759AB"/>
        <Label value="ambulances"              background="#FFA39E"/>
        <Label value="hazard tape"             background="#D4380D"/>
        <Label value="cones"                   background="#FFC069"/>
      </BrushLabels>
    </View>
   ```

   </details>

   > **Warning:** Rectangle labels are mandatory even if you only use segmentation. The results will be segmentation masks but you still need the RectangleLabel values present.

4. **Connect the ML backend**:

   * Go to **Project Settings** → **Model**
   * Click **Connect Model**
   * Enter **URL:** `http://localhost:9090`
   * Enable **Interactive pre‑annotations**

   ![ML Backend Settings](https://github.com/user-attachments/assets/d20c9ee8-64b5-4ee5-b080-0cf0870ef22d)

5. **Enable Auto‑annotation** on any image:

   * Open an image in the labeling interface
   * Toggle **Auto‑annotation** at the bottom
   * (Optional) Add or modify the prompt in the text area for custom queries and pick the corresponding label from below

   ![Auto Annotation](https://github.com/user-attachments/assets/7403735c-c104-4119-b891-7abab6d91551)
   ![Custom Prompt](https://github.com/user-attachments/assets/9828ed8d-2a10-4e43-a8e5-c77046fe591d)

---

## 🛠️ Troubleshooting & Tips

* To update label colors or values, edit the `<Label>` entries in the template.
* Check container health:

  ```bash
  docker ps | grep grounding_sam
  ```
  
* In case label-studio won't load this usually does the trick, also refresh port-forwarding (via VS Code Port tab)

  ```bash
  pkill -f label-studio
  ```
