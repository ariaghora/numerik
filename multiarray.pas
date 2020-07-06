unit multiarray;

{$mode objfpc}
{$H+}
{$modeSwitch advancedRecords}

interface

uses
  Classes, SysUtils, math;

type
  TLongVector = array of longint;
  TSingleVector = array of single;
  TUFunc = function(a: single): single;
  TBFunc = function(a, b: single): single;

type
  generic TTensor<_T> = record
  private
    FIsContiguous: boolean;
    function GetNDims: integer;
    function GetSize: longint;
    function IndexToStridedOffset(Index: array of longint): longint;
    function OffsetToStrided(Offset: longint): longint;
  public
    Data:  array of _T;
    Shape: TLongVector;
    Strides: TLongVector;
    function Contiguous: TTensor;
    function Copy: TTensor;
    function Get(i: longint): _T; overload;
    function Get(i: array of integer): _T; overload;
    function Reshape(NewShape: array of longint): TTensor;
    function T: TTensor;
    procedure Put(i: array of integer; x: _T);
    property IsContiguous: boolean read FIsContiguous write FIsContiguous;
    property NDims: integer read GetNDims;
    property Size: longint read GetSize;
    property Items[i: array of longint]: _T read Get write Put; default;
  end;

  TMultiArray = specialize TTensor<single>;

  TBroadcastResult = record
    A, B: TMultiArray;
  end;


  function AllocateMultiArray(Size: longint): TMultiArray;
  function BroadcastArrays(A, B: TMultiArray): TBroadcastResult;
  function CreateEmptyFTensor(Contiguous: boolean = True): TMultiArray;
  function CreateMultiArray(AData: array of single): TMultiArray;
  function CreateMultiArray(AData: single): TMultiArray;
  function AsStrided(A: TMultiArray; Shape, Strides: array of longint): TMultiArray;
  function ApplyBFunc(A, B: TMultiArray; BFunc: TBFunc): TMultiArray;
  function ApplyUFunc(A: TMultiArray; UFunc: TUFunc): TMultiArray;
  function DynArrayToVector(A: array of longint): TLongVector;
  function ShapeToStrides(AShape: array of longint): TLongVector;

  procedure PrintMultiArray(A: TMultiArray);

  generic function CopyVector<T>(v: T): T;
  generic function Prod<T>(Data: array of T): T;
  generic function Reverse<T>(Data: T): T;
  generic function VectorsEqual<T>(A, B: T): boolean;
  generic procedure PrintVector<T>(Data: T);

  { Arithmetic function wrappers }
  function Add(a, b: single): single;
  function Multiply(a, b: single): single;
  function Subtract(a, b: single): single;

  operator + (A, B: TMultiArray) C: TMultiArray;
  operator * (A, B: TMultiArray) C: TMultiArray;
  operator := (A: single) B: TMultiArray;
  operator explicit(A: single) B: TMultiArray;

implementation

  function AllocateMultiArray(Size: longint): TMultiArray;
  begin
    SetLength(Result.Data, Size);
    SetLength(Result.Shape, 1);
    Result.IsContiguous := True;
    Result.Shape[0] := Size;
    Result.Strides := ShapeToStrides(Result.Shape);
  end;

  function BroadcastArrays(A, B: TMultiArray): TBroadcastResult;
  var
    AdjShapeA, AdjShapeB, AdjStridesA, AdjStridesB: TLongVector;
    i, NewNDims: integer;
  begin
    NewNDims := Max(A.NDims, B.NDims);
    AdjShapeA := specialize Reverse<TLongVector>(A.Shape);
    AdjShapeB := specialize Reverse<TLongVector>(B.Shape);
    AdjStridesA := specialize Reverse<TLongVector>(A.Strides);
    AdjStridesB := specialize Reverse<TLongVector>(B.Strides);

    { Broadcast dimension is always max(A.NDims, B.NDims) }
    if A.NDims < NewNDims then
    begin
      SetLength(AdjShapeA, NewNDims);
      SetLength(AdjStridesA, NewNDims)
    end
    else
    begin
      SetLength(AdjShapeB, NewNDims);
      SetLength(AdjStridesB, NewNDims)
    end;

    for i := 0 to NewNDims - 1 do
      if (AdjShapeA[i] <> AdjShapeB[i]) then
      begin
        if (AdjShapeA[i] < AdjShapeB[i]) and (AdjShapeA[i] <= 1) then
        begin
          AdjStridesA[i] := 0;
          AdjShapeA[i] := AdjShapeB[i]
        end
        else if (AdjShapeB[i] < AdjShapeA[i]) and (AdjShapeB[i] <= 1) then
        begin
          AdjStridesB[i] := 0;
          AdjShapeB[i] := AdjShapeA[i]
        end
        else
          raise Exception.Create('Broadcasting failed.');
      end;

    Result.A := AsStrided(A, specialize Reverse<TLongVector>(AdjShapeA),
                             specialize Reverse<TLongVector>(AdjStridesA));
    Result.B := AsStrided(B, specialize Reverse<TLongVector>(AdjShapeB),
                             specialize Reverse<TLongVector>(AdjStridesB));
  end;

  function CreateEmptyFTensor(Contiguous: boolean = True):TMultiArray;
  begin
    Result.IsContiguous := Contiguous;
  end;

  function CreateMultiArray(AData: array of single): TMultiArray;
  var
    i: longint;
  begin
    Result := CreateEmptyFTensor(True);
    SetLength(Result.Data, Length(AData));
    for i := 0 to Length(AData) - 1 do
      Result.Data[i] := AData[i];
    SetLength(Result.Shape, 1);
    Result.Shape[0] := Length(AData);
    Result.Strides := ShapeToStrides(Result.Shape);
  end;

  function CreateMultiArray(AData: single): TMultiArray;
  begin
    Result := CreateMultiArray([AData]);
  end;

  function AsStrided(A: TMultiArray; Shape, Strides: array of longint): TMultiArray;
  begin
    Result := CreateEmptyFTensor(false);
    Result.Data := A.Data;
    Result.Shape := DynArrayToVector(Shape);
    Result.Strides := DynArrayToVector(Strides);
  end;

  function ApplyBFunc(A, B: TMultiArray; BFunc: TBFunc): TMultiArray;
  var
    i: longint;
    BcastResult: TBroadcastResult;
  begin
    if not(specialize VectorsEqual<TLongVector>(A.Shape, B.Shape)) then
    begin
      BcastResult := BroadcastArrays(A, B);
      A := BcastResult.A;
      B := BcastResult.B;
    end;

    Result := CreateEmptyFTensor(True);
    SetLength(Result.Data, A.Size);
    Result.Shape := A.Shape;
    Result := Result.Reshape(A.Shape); // should be reset strides
    for i := 0 to A.Size - 1 do
      Result.Data[i] := BFunc(A.Get(i), B.Get(i));
  end;

  function ApplyUFunc(A: TMultiArray; UFunc: TUFunc): TMultiArray;
  var
    i: longint;
  begin
    Result := A.Copy();
    for i := 0 to High(A.Data) do
      Result.Data[i] := UFunc(Result.Data[i]);
  end;

  function DynArrayToVector(A: array of longint): TLongVector;
  var
    i: longint;
  begin
    Result := nil;
    SetLength(Result, Length(A));
    for i := 0 to High(A) do
      Result[i] := A[i];
  end;

  procedure PrintMultiArray(A: TMultiArray);
  var
    r, c, i: longint;
  begin
    if A.NDims = 1 then
    begin
      for i := 0 to A.Size - 1 do
        Write(A.Get(i), ' ');
      WriteLn;
    end;

    if A.NDims = 2 then
    begin
      for r := 0 to A.Shape[0] - 1 do
      begin
        for c := 0 to A.Shape[1] - 1 do
        begin
          Write(A.Get([r, c]) : 5 : 2);
          if c < A.Shape[1] - 1 then Write(' ');
        end;
        WriteLn;
      end;
    end;
  end;

  generic function CopyVector<T>(v: T): T;
  var
    i: longint;
  begin
    SetLength(Result, Length(v));
    for i := 0 to High(v) do
      Result[i] := v[i];
  end;

  generic function Prod<T>(Data: array of T): T;
  var
    x: T;
  begin
    Result := 1;
    for x in Data do
      Result := Result * x;
  end;

  generic function Reverse<T>(Data: T): T;
  var
    i: longint;
  begin
    Result := nil;
    SetLength(Result, Length(Data));
    For i := High(Data) downto 0 do
      Result[High(Data) - i] := Data[i];
  end;

  generic function VectorsEqual<T>(A, B: T): boolean;
  var
    i: longint;
  begin
    if (Length(A) <> Length(B)) then Exit(false);
    Result := True;
    for i := 0 to High(A) do
      Result := Result and (A[i] = B[i]);
  end;

  generic procedure PrintVector<T>(Data: T);
  var
    i: longint;
  begin
    for i := 0 to Length(Data) - 1 do
    begin
      Write(Data[i]);
      if i < Length(Data) - 1 then
        Write(', ');
    end;
    WriteLn;
  end;

  function ShapeToStrides(AShape: array of longint): TLongVector;
  var
    k, j, prod: integer;
  begin
    Result := nil;

    SetLength(Result, Length(AShape));
    for k := 0 to Length(Result) - 1 do
    begin
      prod := 1;
      for j := k + 1 to Length(Result) - 1 do
        prod := prod * AShape[j];
      Result[k] := prod;
    end;
  end;

  generic function TTensor<_T>.OffsetToStrided(Offset: longint): longint;
  var
    r, c: longint;
  begin
    if NDims > 2 then
      raise ENotImplemented.Create('Item access with not implemented yet with NDims > 2.');

    if NDims = 1 then Exit(Offset);
    if NDims = 2 then
    begin
      r := Offset div Shape[1];
      c := Offset mod Shape[1];
      Exit(IndexToStridedOffset([r, c]));
    end;
  end;

  generic function TTensor<_T>.GetNDims: integer;
  begin
    Exit(Length(Shape));
  end;

  generic function TTensor<_T>.GetSize: longint;
  begin
    Exit(specialize Prod<longint>(Shape));
  end;

  function TTensor.IndexToStridedOffset(Index: array of longint): longint;
  var
    i: integer;
  begin
    Result := 0;
    for i := 0 to Length(Index) - 1 do
      Result := Result + (Strides[i] * Index[i]);
  end;

  function TTensor.Contiguous: TTensor;
  begin
    if IsContiguous then Exit(self);
  end;

  function TTensor.Copy: TTensor;
  var
    i: longint;
  begin
    Result.IsContiguous := self.IsContiguous;
    SetLength(Result.Data, length(Self.Data));
    for i := 0 to High(self.Data) do
      Result.Data[i] := self.Data[i];
    Result.Shape := specialize CopyVector<TLongVector>(self.Shape);
    Result.Strides := specialize CopyVector<TLongVector>(self.Strides);
  end;

  function TTensor.Get(i: longint): single; overload;
  begin
    if IsContiguous then Exit(Data[i]);
    Exit(Data[OffsetToStrided(i)]);
  end;

  function TTensor.Get(i: array of integer): single; overload;
  begin
    Assert(length(i) = length(Shape), 'Index out of bounds.');
    Exit(Data[IndexToStridedOffset(i)]);
  end;

  function TTensor.Reshape(NewShape: array of longint): TTensor;
  var
    i: integer;
  begin
    Assert((specialize Prod<longint>(NewShape)) = (specialize Prod<longint>(Shape)), 'Impossible reshape.');
    Result.IsContiguous := False;
    Result.Data := self.Data;
    SetLength(Result.Shape, Length(NewShape));
    for i := 0 to Length(NewShape) - 1 do
      Result.Shape[i] := NewShape[i];
    Result.Strides := ShapeToStrides(Result.Shape);
  end;

  function TTensor.T: TTensor;
  begin
    Result := Self.Copy();
    Result.Shape := specialize Reverse<TLongVector>(Self.Shape);
    Result.Strides := specialize Reverse<TLongVector>(Self.Strides);
  end;

  procedure TTensor.Put(i: array of integer; x: _T);
  begin
    Data[Self.IndexToStridedOffset(i)] := x;
  end;

  function Add(a, b: single): single;
  begin
    Exit(a + b);
  end;

  function Multiply(a, b: single): single;
  begin
    Exit(a * b);
  end;

  function Subtract(a, b: single): single;
  begin
    Exit(a - b);
  end;

  operator +(A, B: TMultiArray) C: TMultiArray;
  begin
    C := ApplyBFunc(A, B, @Add);
  end;

  operator *(A, B: TMultiArray) C: TMultiArray;
  begin
    C := ApplyBFunc(A, B, @Multiply);
  end;

  operator :=(A: single) B: TMultiArray;
  begin
    B := CreateMultiArray(A);
  end;

  operator explicit(A: single) B: TMultiArray;
  begin
    B := CreateMultiArray(A);
  end;

end.

