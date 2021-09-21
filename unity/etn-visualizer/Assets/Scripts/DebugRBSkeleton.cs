using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Security;
using UnityEngine;

public class DebugRBSkeleton : MonoBehaviour
{
    [SerializeField] private int framerate;

    private float timer;
    private float frameTime;

    public TMPro.TextMeshProUGUI debugUI;

    // private Dictionary<string, Dictionary<string, Transform>> skeletonList;
    private Dictionary<string, List<Transform>> rigList;

    private char separator = ';'; // must match with python-side separator.

    private void Awake()
    {
        frameTime = 1f / framerate;
        timer = 0;
        // skeletonList = new Dictionary<string, Dictionary<string, Transform>>();
        rigList = new Dictionary<string, List<Transform>>();
    }

    void Update()
    {
        while (PythonLauncher.data.Count > 0)
        {
            if (timer > 0)
            {
                timer -= Time.deltaTime;
                return;
            }
            try
            {
                // Try consume data
                HandlePythonData(PythonLauncher.data[0]);
                PythonLauncher.data.RemoveAt(0);
            }
            catch (Exception e)
            {
                Debug.LogWarning("Something went wrong with: " + PythonLauncher.data[0] + "\n" + e.Message + "\n" + e.StackTrace);
                PythonLauncher.data.RemoveAt(0);
                throw e;
            }
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
            case "P": // Animation pose
                ProcessPoseData(messageSplit);
                timer = frameTime;
                break;
            case "A": // Additional anim data (per frame)
                ProcessAddtionalData(messageSplit);
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
        
        GameObject rigParent = GameObject.Find($@"model_{rigName}");

        List<Transform> rigJointList = new List<Transform>();
        
        if (rigParent == null)
        {
            Debug.LogError($"HIERARCHY PARSE ERROR: Couldn't find model for rig-name '{rigName}' ");
            return;
        }
        
        foreach (string jointName in jointList)
        {
            var childList = rigParent.GetComponentsInChildren<Transform>();
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
        Debug.Log("Registered hierarchy: " + rigName);
        
    }
    
    void ProcessPoseData(string[] unparsedData)
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
        Vector3 root_pos = new Vector3(pose[0], pose[1], pose[2]);
        
        ArraySegment<float> poseQuats = new ArraySegment<float>(pose, 3, pose.Length - 3);
        
        //Set root node position.
        rig[0].SetPositionAndRotation(root_pos, Quaternion.identity);
        
        //set joint rotations
        for (int i = 0; i < rig.Count; i++)
        {
            //Parse quat
            Quaternion jointRot = new Quaternion(
                poseQuats.Array[i * 4],
                poseQuats.Array[i * 4 + 1],
                poseQuats.Array[i * 4 + 2],
                poseQuats.Array[i * 4 + 3]
            );
            
            //Lafan Data is in another coordinate system so we have to convert stuff first.
            jointRot = Convert_quat_xzy_to_xyz(jointRot);
            
            rig[i].localRotation = jointRot;
        }
        
        
    }

    Quaternion Convert_quat_xzy_to_xyz(Quaternion quatIn)
    {
        Quaternion quatOut = new Quaternion(
            quatIn.x, //TODO: QUAT CONVERSION
            quatIn.y,
            quatIn.z,
            quatIn.w
        );
        
        // quatIn.ToAngleAxis(out float angle, out Vector3 axis);
        // Vector3 newAxis = new Vector3(axis.x, -axis.y, -axis.z);
        // Quaternion quatOut = Quaternion.AngleAxis(angle, newAxis);
        return quatOut;
    }

    void ProcessAddtionalData(string[] unparsedData)
    {
        //Temp, just print info into textbox
        string parsedText = unparsedData[1].Replace(separator, '\n');
        debugUI.SetText(parsedText);
    }
}
