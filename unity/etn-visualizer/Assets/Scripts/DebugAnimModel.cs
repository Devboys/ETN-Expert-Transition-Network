using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Security;
using UnityEngine;

public class DebugAnimModel : MonoBehaviour
{
    [SerializeField] private int framerate;
    [Tooltip("If enabled, will force root joint to position (0,0,0) in every frame.")]
    [SerializeField] private bool lockRootPos;

    [SerializeField] private GameObject jointTemplate;

    public bool playContinuous = false;

    private float timer;

    public TMPro.TextMeshProUGUI debugUI;
    public TMPro.TextMeshProUGUI debugUITwo;

    // private Dictionary<string, Dictionary<string, Transform>> skeletonList;
    private Dictionary<string, List<Transform>> rigList; //holds models
    private Dictionary<string, List<Transform>> generatedRigList;  // holds generated hierarchies.
    private Dictionary<string, Vector3> firstRootPos;

    private char separator = ';'; // must match with python-side separator.
    

    void Awake()
    {
        timer = 0;
        rigList = new Dictionary<string, List<Transform>>();
        generatedRigList = new Dictionary<string, List<Transform>>();
        firstRootPos = new Dictionary<string, Vector3>();
    }

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Keypad7))
        {
            playContinuous = !playContinuous;
        }
        if (Input.GetKeyDown(KeyCode.Keypad8))
        {
            lockRootPos = !lockRootPos;
        }

        if (timer > 0)
        {
            timer -= Time.deltaTime;
            return;
        }

        if (PythonLauncher.instance.data.Count > 0)
        {
            try
            {
                // Try consume data
                HandlePythonData(PythonLauncher.instance.data[0]);
                PythonLauncher.instance.data.RemoveAt(0);
            }
            catch (Exception e)
            {
                Debug.LogWarning("Something went wrong with: \"" + PythonLauncher.instance.data[0] + "\"\n" + e.Message + "\n" + e.StackTrace);
                PythonLauncher.instance.data.RemoveAt(0);
                throw e;
            }
        }
        else if(playContinuous)
        {
            PythonLauncher.instance.WriteToProcess("NAV;NEXT");
        }
    }

    

    void HandlePythonData(string message)
    {
        string[] messageSplit = message.Split(' ');
        string marker = messageSplit[0]; // Marker should always be first part

        switch (marker)
        {
            case "H": // Hierarchy definition
                ProcessHierarchyDefinition(messageSplit);
                break;
            case "P": // Animation pose, rotations
                ProcessPoseData_Quats(messageSplit);
                break;
            case "G": //Animation pose, global positions
                ProcessPoseData_Positions(messageSplit);
                break;
            case "A": // Additional anim data (per frame)
                ProcessAddtionalData(messageSplit);
                break;
            case "E": //end-of-frame info
                float fr = framerate > 0 ? framerate : 1; //at least 1
                timer = 1f / fr;
                break;
            default:
                Debug.LogError($"MESSAGE PARSE ERROR: Unhandled marker: '{marker}'");
                break;
        }
    }
    
    void ProcessHierarchyDefinition(string[] unparsedData)
    {
        string rigName = unparsedData[1];
        string[] jointList = unparsedData[2].Split(separator);
        int[] parentList = Array.ConvertAll(unparsedData[3].Split(separator), int.Parse);
        
        Transform rootParent = GameObject.Find($@"model_{rigName}").transform;
        
        GenerateRigFromDefinition(rigName, jointList, parentList, rootParent);
        RegisterModelRig(rigName, jointList, rootParent);
        
        Debug.Log("Registered hierarchy: " + rigName);
    }

    void ProcessPoseData_Quats(string[] unparsedData)
    {
        string rigName = unparsedData[1];
        List<Transform> rig;
        if (!rigList.TryGetValue(rigName, out rig))
        {
            Debug.LogError($"POSE PROCESS ERROR: Couldn't find rig with name: '{rigName}'");
            return;
        }

        string[] poseUnparsed = unparsedData[2].Split(separator);
        float[] pose = Array.ConvertAll(poseUnparsed, float.Parse);
        
        //split pose into rootpos + quats. Quat indices in data are the same as indices in the parsed hierarchy. 
        Vector3 root_pos = new Vector3(-pose[0],pose[1], pose[2] );
        if (lockRootPos)
        {
            root_pos = Vector3.zero;
        }

        //make animation start at initial GO position
        if (!firstRootPos.TryGetValue(rigName, out Vector3 firstRoot))
        {
            firstRootPos.Add(rigName, root_pos);
            firstRoot = root_pos;
        }
        root_pos -= firstRoot;

        //Set root node position.
        rig[0].localPosition = root_pos;
        
        //set joint rotations
        for (int i = 0; i < rig.Count; i++)
        {
            Quaternion parsedQuat = new Quaternion(
                pose[3 + i * 4 + 1],
                pose[3 + i * 4 + 2],
                pose[3 + i * 4 + 3],
                pose[3 + i * 4 + 0]
            );
            
            Quaternion jointQuat = leftToRightCoord(parsedQuat);

            rig[i].localRotation = jointQuat;
        }
    }

    void ProcessPoseData_Positions(string[] unparsedData)
    {
        string rigName = unparsedData[1];
        List<Transform> rig;
        if (!generatedRigList.TryGetValue(rigName, out rig))
        {
            Debug.LogError($"POSE PROCESS ERROR: Couldn't find generated rig with name: '{rigName}'");
            return;
        }

        string[] poseUnparsed = unparsedData[2].Split(separator);
        float[] pose = Array.ConvertAll(poseUnparsed, float.Parse);

        //set joint positions
        for (int i = 0; i < rig.Count; i++)
        {
            Vector3 jointPos = new Vector3(
                -pose[i * 3 + 0],
                pose[i * 3 + 1],
                pose[i * 3 + 2]
            );
            
            if (lockRootPos)
            {
                jointPos -= new Vector3(-pose[0], pose[1], pose[2]);
            }
            
            //make animation start at initial GO position
            if (!firstRootPos.TryGetValue(rigName, out Vector3 firstRoot))
            {
                firstRootPos.Add(rigName, jointPos);
                firstRoot = jointPos;
            }
            jointPos -= firstRoot;

            rig[i].position = jointPos + rig[0].parent.position; //Note, this is global position
            
            
        }
    }

    void ProcessAddtionalData(string[] unparsedData)
    {
        //Temp, just print info into textbox
        string rigName = unparsedData[1];
        int frameIndex = int.Parse(unparsedData[2]);
        string parsedText = unparsedData[3].Replace(separator, '\n');

        if (rigName == "original")
        {
            debugUI.SetText(parsedText);
        }
        else
        {
            debugUITwo.SetText(parsedText);
        }

        PythonLauncher.instance.WriteToProcess("FRAME;" + frameIndex);
    }
    
    Quaternion leftToRightCoord(Quaternion quatIn)
    {
        Quaternion quatOut = new Quaternion(
            quatIn.x,
            -quatIn.y,
            -quatIn.z,
            quatIn.w
        );
        
        return quatOut;
    }
    
    /// <summary>
    /// Generates an unskinned skeleton rig from the provided hierarchy definition.
    /// </summary>
    /// <param name="rigName"> The name of the rig. Used for dictionary lookup when processing pose data</param>
    /// <param name="jointList"> The list of joint names in the hierarchy. Order is expected to mirror rig hierarchy parsing order</param>
    /// <param name="parentList"> The list of parent-indices per joint. Each element contains the parent index of the joint at the same element-index. </param>
    void GenerateRigFromDefinition(string rigName, string[] jointList, int[] parentList, Transform rootParent)
    {
        List<Transform> transformList = new List<Transform>();

        // Create joints from templates
        for (int j = 0; j < jointList.Length; j++)
        {
            GameObject jointGO = Instantiate(jointTemplate);
            jointGO.name = "Gen:" + jointList[j];
            transformList.Add(jointGO.transform);
        }
        
        // Apply parent-child hierarchy
        for (int p = 0; p < parentList.Length; p++)
        {
            int parentID = parentList[p];
            Transform parentTransform = (parentID == -1) ? rootParent : transformList[parentID];
            transformList[p].parent = parentTransform;
        }

        // Save rig
        generatedRigList.Add(rigName, transformList);
    }
    
    /// <summary>
    /// Registers an existing rig from a joint-list hierarchy for the purpose of applying poses.
    /// </summary>
    /// <param name="rigName">The name to identify this rig with. Will attempt to find a model with the name "model_<rigName>" </param>
    /// <param name="jointList">The list of joint names to register in the rig. Order of elements is expected to mirror rig hierarchy parsing order </param>
    void RegisterModelRig(string rigName, string[] jointList, Transform rootParent)
    {
        List<Transform> rigJointList = new List<Transform>();
        
        if (rootParent == null)
        {
            Debug.LogError($"HIERARCHY PARSE ERROR: Couldn't find model for rig-name '{rigName}' ");
            return;
        }
        
        foreach (string jointName in jointList)
        {
            var childList = rootParent.GetComponentsInChildren<Transform>();
            Transform jointTransform = null;

            foreach (Transform childTransform in childList)
            {
                if (childTransform.gameObject.name == $@"Model:{jointName}")
                {
                    jointTransform = childTransform;
                    break;
                }
            }

            if (jointTransform != null)
            {
                rigJointList.Add(jointTransform);
            }
        }
        
        rigList.Add(rigName, rigJointList);
    }
}